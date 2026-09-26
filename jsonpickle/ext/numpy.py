from __future__ import annotations

import ast
import json
import math
import sys
import warnings
import zlib
from collections.abc import Sequence
from types import ModuleType
from typing import Any, NoReturn, Protocol, cast

import numpy as np
import numpy.typing as npt

from ..backend import json as json_backend
from ..handlers import BaseHandler, register, unregister
from ..pickler import Pickler
from ..unpickler import Unpickler
from ..util import b64decode, b64encode, importable_name, loadclass

__all__ = ["register_handlers", "unregister_handlers", "encode_oob", "decode_oob"]

#: envelope tag wrapping the flattened value in out-of-band (oob) metadata
OOB_TAG = "py/numpy_oob"
#: tag marking an out-of-band buffer reference inside an ndarray payload
OOB_BUFFER_TAG = "py/numpy_buffer"


# matches `np.ndarray`, `np.generic`, and anything else with a `.dtype` attribute
class _HasDType(Protocol):
    @property
    def dtype(self) -> np.dtype[Any]: ...


native_byteorder: str = "<" if sys.byteorder == "little" else ">"


def get_byteorder(arr: _HasDType) -> str:
    """translate equals sign to native order"""
    byteorder = arr.dtype.byteorder
    return native_byteorder if byteorder == "=" else byteorder


def _buffer_nbytes(buf: Any) -> int:
    """Return the length of a bytes-like object in bytes"""
    if isinstance(buf, memoryview):
        return buf.nbytes
    return len(buf)


def _normalize_oob_buffer(buf: Any, index: int) -> "bytes | bytearray | memoryview":
    """Validate and normalize one caller-provided out-of-band buffer.

    Accepts bytes, bytearray and memoryview. Contiguous memoryviews are
    used zero-copy (cast to a byte view when needed); non-contiguous
    memoryviews are explicitly copied into a packed C-contiguous bytes
    object. Anything else raises TypeError.
    """
    if isinstance(buf, (bytes, bytearray)):
        return buf
    if isinstance(buf, memoryview):
        if buf.c_contiguous:
            return buf if buf.format == "B" else buf.cast("B")
        # explicit, documented copy: a non-contiguous view has no single
        # underlying byte range to borrow, so pack it into fresh bytes
        return buf.tobytes()
    raise TypeError(
        f"out-of-band buffer {index} must be bytes, bytearray or memoryview, "
        f"got {type(buf).__name__}"
    )


def _validate_buffer_range(
    bufsize: int,
    dtype: np.dtype[Any],
    shape: Any,
    offset: int = 0,
    strides: Any = None,
) -> None:
    """Ensure every byte reachable by a view lies inside its buffer.

    Raises ValueError before numpy is ever handed an out-of-bounds view.
    """
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or offset > bufsize
    ):
        raise ValueError(
            f"view offset {offset!r} is outside a buffer of {bufsize} bytes"
        )
    shape = tuple(shape)
    if strides is not None:
        strides = tuple(strides)
        if len(strides) != len(shape):
            raise ValueError(
                f"view has {len(shape)} dimensions but "
                f"{len(strides)} strides were given"
            )
    if 0 in shape:
        # no element is reachable, so no byte range can be out of bounds
        return
    itemsize = dtype.itemsize
    if strides is None:
        computed = []
        step = itemsize
        for dim in reversed(shape):
            computed.append(step)
            step *= dim
        strides = tuple(reversed(computed))
    lo = hi = offset
    for dim, stride in zip(shape, strides):
        extent = (dim - 1) * stride
        if extent >= 0:
            hi += extent
        else:
            lo += extent
    hi += itemsize
    if lo < 0 or hi > bufsize:
        raise ValueError(
            f"view with shape {shape}, strides {strides}, offset {offset} and "
            f"itemsize {itemsize} reaches byte range [{lo}, {hi}), which is "
            f"outside its buffer of {bufsize} bytes"
        )


def _validate_owned_buffer(bufsize: int, dtype: np.dtype[Any], shape: Any) -> None:
    """Ensure a buffer is large enough for an array owning its data"""
    required = dtype.itemsize * math.prod(shape)
    if bufsize < required:
        raise ValueError(
            f"buffer of {bufsize} bytes is too small for an array of shape "
            f"{tuple(shape)} and dtype {dtype} ({required} bytes required)"
        )


class NumpyBaseHandler(BaseHandler):
    def flatten_dtype(self, dtype: npt.DTypeLike, data: dict[str, Any]) -> None:
        if hasattr(dtype, "tostring"):
            data["dtype"] = dtype.tostring()  # type: ignore[union-attr]
        else:
            dtype = str(dtype)
            prefix = "(numpy.record, "
            if dtype.startswith(prefix):
                dtype = dtype[len(prefix) : -1]
            data["dtype"] = dtype

    def restore_dtype(self, data: dict[str, Any]) -> np.dtype[Any]:
        dtype = data["dtype"]
        if dtype.startswith(("{", "[")):
            dtype = ast.literal_eval(dtype)
        return np.dtype(dtype)  # type: ignore[no-any-return]


class NumpyDTypeHandler(NumpyBaseHandler):
    def flatten(self, obj: npt.DTypeLike, data: dict[str, Any]) -> dict[str, Any]:
        self.flatten_dtype(obj, data)
        return data

    def restore(self, data: dict[str, Any]) -> npt.DTypeLike:
        return self.restore_dtype(data)


class NumpyGenericHandler(NumpyBaseHandler):
    def flatten(self, obj: npt.NDArray[Any], data: dict[str, Any]) -> dict[str, Any]:
        self.flatten_dtype(obj.dtype.newbyteorder("N"), data)
        value = obj.tolist()
        if isinstance(value, np.generic):
            # extended-precision scalars have no Python counterpart, so
            # tolist() hands back the scalar itself; store its text form
            value = str(obj)
        data["value"] = self.context.flatten(value, reset=False)
        return data

    def restore(self, data: dict[str, Any]) -> Any:
        value = self.context.restore(data["value"], reset=False)
        # rebuild through the dtype, not dtype.type(value): the scalar
        # constructor ignores everything the dtype carries beyond its kind,
        # such as the datetime64/timedelta64 unit and the field layout of a
        # structured scalar
        return np.array(value, dtype=self.restore_dtype(data))[()]


class NumpyDatetimeHandler(NumpyGenericHandler):
    """Retained for backwards compatibility; datetime64 needs no special case"""


class UnpickleableNumpyGenericHandler(NumpyGenericHandler):
    """
    From issue #381, this is used for simplifying the output of numpy arrays
    when unpicklable=False (the default is True).
    """

    # TODO: narrow return value down from Any
    def flatten(self, obj: npt.NDArray[Any], data: dict[str, Any]) -> Any:
        if not self.context.unpicklable:
            return self.context.flatten(obj.tolist(), reset=False)
        else:
            return super(NumpyGenericHandler, self).flatten(obj, data)

    def restore(self, data: dict[str, Any]) -> NoReturn:
        raise NotImplementedError


class NumpyNDArrayHandler(NumpyBaseHandler):
    """Stores arrays as text representation, without regard for views"""

    def flatten_flags(self, obj: npt.NDArray[Any], data: dict[str, Any]) -> None:
        if obj.flags.writeable is False:
            data["writeable"] = False

    def restore_flags(self, data: dict[str, Any], arr: npt.NDArray[Any]) -> None:
        if not data.get("writeable", True):
            arr.flags.writeable = False

    def flatten(self, obj: npt.NDArray[Any], data: dict[str, Any]) -> dict[str, Any]:
        self.flatten_dtype(obj.dtype.newbyteorder("N"), data)
        self.flatten_flags(obj, data)
        data["values"] = self.context.flatten(obj.tolist(), reset=False)
        if 0 in obj.shape:
            # add shape information explicitly as it cannot be
            # inferred from an empty list
            data["shape"] = obj.shape
        return data

    def restore(self, data: dict[str, Any]) -> npt.NDArray[Any]:
        values = self.context.restore(data["values"], reset=False)
        arr = np.array(
            values, dtype=self.restore_dtype(data), order=data.get("order", "C")
        )
        shape = data.get("shape", None)
        if shape is not None:
            arr = arr.reshape(shape)

        self.restore_flags(data, arr)
        return arr


class NumpyNDArrayHandlerBinary(NumpyNDArrayHandler):
    """stores arrays with size greater than 'size_threshold' as
    (optionally) compressed base64

    Notes
    -----
    This would be easier to implement using np.save/np.load, but
    that would be less language-agnostic
    """

    def __init__(
        self, size_threshold: int = 16, compression: ModuleType = zlib
    ) -> None:
        """
        :param size_threshold: nonnegative int or None
            valid values for 'size_threshold' are all nonnegative
            integers and None
            if size_threshold is None, values are always stored as nested lists
        :param compression: a compression module or None
            valid values for 'compression' are {zlib, bz2, None}
            if compression is None, no compression is applied
        """
        self.size_threshold = size_threshold
        self.compression = compression

    def flatten_byteorder(self, obj: npt.NDArray[Any], data: dict[str, Any]) -> None:
        byteorder = obj.dtype.byteorder
        if byteorder != "|":
            data["byteorder"] = get_byteorder(obj)

    def restore_byteorder(self, data: dict[str, Any], arr: npt.NDArray[Any]) -> None:
        byteorder = data.get("byteorder", None)
        if byteorder:
            arr.dtype = arr.dtype.newbyteorder(byteorder)  # type: ignore[misc]

    def flatten(self, obj: npt.NDArray[Any], data: dict[str, Any]) -> dict[str, Any]:
        """encode numpy to json"""
        if self.size_threshold is None or self.size_threshold >= obj.size:
            # encode as text
            data = super().flatten(obj, data)
        else:
            # encode as binary
            if obj.dtype == object:
                # There's a bug deep in the bowels of numpy that causes a
                # segfault when round-tripping an ndarray of dtype object.
                # E.g., the following will result in a segfault:
                #     import numpy as np
                #     arr = np.array([str(i) for i in range(3)],
                #                    dtype=object)
                #     dtype = arr.dtype
                #     shape = arr.shape
                #     buf = arr.tobytes()
                #     del arr
                #     arr = np.ndarray(buffer=buf, dtype=dtype,
                #                      shape=shape).copy()
                # So, save as a binary-encoded list in this case
                buf = json.dumps(obj.tolist()).encode()
            elif hasattr(obj, "tobytes"):
                # numpy docstring is lacking as of 1.11.2,
                # but this is the option we need
                buf = obj.tobytes(order="A")
            else:
                # numpy < 1.9 compatibility
                buf = obj.tostring(order="a")  # type: ignore[attr-defined]
            if self.compression:
                buf = self.compression.compress(buf)
            oob_buffers = getattr(self.context, "numpy_oob_buffers", None)
            if oob_buffers is not None:
                # out-of-band mode: ship the buffer side-channel and keep
                # only its index in the JSON metadata
                data["values"] = {OOB_BUFFER_TAG: len(oob_buffers)}
                oob_buffers.append(buf)
            else:
                data["values"] = b64encode(buf)
            data["shape"] = obj.shape
            self.flatten_dtype(obj.dtype.newbyteorder("N"), data)
            self.flatten_byteorder(obj, data)
            self.flatten_flags(obj, data)

            if not obj.flags.c_contiguous:
                data["order"] = "F"

        return data

    def restore(self, data: dict[str, Any]) -> npt.NDArray[Any]:
        """decode numpy from json"""
        values = data["values"]
        if isinstance(values, list):
            # decode text representation
            arr = super().restore(data)
        elif isinstance(values, (int, float)):
            # single-value array
            arr = np.array([values], dtype=self.restore_dtype(data))
        else:
            # decode binary representation
            dtype = self.restore_dtype(data)
            if isinstance(values, dict):
                if OOB_BUFFER_TAG not in values:
                    raise ValueError("invalid binary ndarray payload")
                buf = self._restore_oob_buffer(values[OOB_BUFFER_TAG])
            else:
                buf = b64decode(values)
            if self.compression:
                buf = self.compression.decompress(buf)
            # See note above about segfault bug for numpy dtype object. Those
            # are saved as a list to work around that.
            if dtype == object:
                values = json.loads(bytes(buf).decode())
                arr = np.array(values, dtype=dtype, order=data.get("order", "C"))
                shape = data.get("shape", None)
                if shape is not None:
                    arr = arr.reshape(shape)
            else:
                shape = data.get("shape")
                if shape is not None:
                    _validate_owned_buffer(_buffer_nbytes(buf), dtype, shape)
                arr = np.ndarray(
                    buffer=buf,
                    dtype=dtype,
                    shape=shape,  # type: ignore[arg-type]
                    order=data.get("order", "C"),
                ).copy()  # make a copy, to force the result to own the data
                self.restore_byteorder(data, arr)
            self.restore_flags(data, arr)

        return arr

    def _restore_oob_buffer(self, index: Any) -> "bytes | bytearray | memoryview":
        """Fetch one caller-provided out-of-band buffer by index"""
        oob_buffers: "list[bytes | bytearray | memoryview] | None" = getattr(
            self.context, "numpy_oob_buffers", None
        )
        if oob_buffers is None:
            raise ValueError(
                "metadata references an out-of-band numpy buffer; decode it "
                "with jsonpickle.ext.numpy.decode_oob() and provide the "
                "matching buffers"
            )
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(oob_buffers)
        ):
            raise ValueError(
                f"out-of-band buffer index {index!r} is out of range for "
                f"{len(oob_buffers)} provided buffer(s)"
            )
        return oob_buffers[index]


class NumpyNDArrayHandlerView(NumpyNDArrayHandlerBinary):
    """Pickles references inside ndarrays, or array-views

    Notes
    -----
    The current implementation has some restrictions.

    'base' arrays, or arrays which are viewed by other arrays,
    must be f-or-c-contiguous.
    This is not such a large restriction in practice, because all
    numpy array creation is c-contiguous by default.
    Relaxing this restriction would be nice though; especially if
    it can be done without bloating the design too much.

    Furthermore, ndarrays which are views of array-like objects
    implementing __array_interface__,
    but which are not themselves nd-arrays, are deepcopied with
    a warning (by default),
    as we cannot guarantee whatever custom logic such classes
    implement is correctly reproduced.
    """

    def __init__(
        self,
        mode: str = "warn",
        size_threshold: int = 16,
        compression: ModuleType = zlib,
    ) -> None:
        """
        :param mode: {'warn', 'raise', 'ignore'}
            How to react when encountering array-like objects whose
            references we cannot safely serialize
        :param size_threshold: nonnegative int or None
            valid values for 'size_threshold' are all nonnegative
            integers and None
            if size_threshold is None, values are always stored as nested lists
        :param compression: a compression module or None
            valid values for 'compression' are {zlib, bz2, None}
            if compression is None, no compression is applied
        """
        super().__init__(size_threshold, compression)
        self.mode = mode

    def _flatten_values_for_readability(
        self, obj: npt.NDArray[Any], data: dict[str, Any]
    ) -> None:
        """Add a human-readable values entry without touching the
        reference bookkeeping.

        When an array is stored by reference to a base, the values entry is
        purely decorative: restore reconstructs the array from base and
        never reads it back. Flattening the values through self.context
        would register additional py/id references that the restore step
        does not mirror, desynchronizing the id counters and making later
        py/id references resolve to the wrong object (see #449). Snapshot
        and restore the reference state so the decorative flatten is a no-op as
        far as py/id allocation is concerned.
        """
        context = self.context
        saved_objs = dict(context._objs)
        saved_flattened = dict(context._flattened)
        super(NumpyNDArrayHandlerBinary, self).flatten(obj, data)
        context._objs = saved_objs
        context._flattened = saved_flattened

    def flatten(self, obj: npt.NDArray[Any], data: dict[str, Any]) -> dict[str, Any]:
        """encode numpy to json"""
        base = obj.base
        deepcopy_failed = False
        if base is None and obj.flags.forc:
            # store by value
            data = super().flatten(obj, data)
            # ensure that views on arrays stored as text
            # are interpreted correctly
            if not obj.flags.c_contiguous:
                data["order"] = "F"
        elif isinstance(base, np.ndarray) and base.flags.forc:
            # store by reference
            data["base"] = self.context.flatten(base, reset=False)

            offset = obj.ctypes.data - base.ctypes.data
            if offset:
                data["offset"] = offset

            if not obj.flags.c_contiguous:
                data["strides"] = obj.strides

            data["shape"] = obj.shape
            self.flatten_dtype(obj.dtype.newbyteorder("N"), data)
            self.flatten_flags(obj, data)

            if get_byteorder(obj) != "|":
                byteorder = "S" if get_byteorder(obj) != get_byteorder(base) else None
                if byteorder:
                    data["byteorder"] = byteorder

            if self.size_threshold is None or self.size_threshold >= obj.size:
                # not used in restore since base is present, but
                # include values for human-readability
                self._flatten_values_for_readability(obj, data)
        elif base is not None:
            try:
                base_buf = np.frombuffer(base, dtype=np.uint8)
            except (TypeError, ValueError):
                base_buf = None

            allow_buffer_base = False
            if base_buf is not None and base_buf.flags.forc:
                if isinstance(base, (bytes, bytearray)):
                    allow_buffer_base = obj.dtype.fields is not None
                else:
                    allow_buffer_base = True

            if allow_buffer_base:
                assert base_buf is not None
                if isinstance(base, (bytes, bytearray)):
                    base_store = base
                elif isinstance(base, memoryview):
                    base_store = base.tobytes()
                else:
                    base_store = base_buf.tobytes()

                data["base"] = self.context.flatten(base_store, reset=False)

                offset = obj.ctypes.data - base_buf.ctypes.data
                if offset:
                    data["offset"] = offset

                if not obj.flags.c_contiguous:
                    data["strides"] = obj.strides

                data["shape"] = obj.shape
                self.flatten_dtype(obj.dtype.newbyteorder("N"), data)
                self.flatten_flags(obj, data)

                if get_byteorder(obj) != "|":
                    data["byteorder"] = get_byteorder(obj)

                if self.size_threshold is None or self.size_threshold >= obj.size:
                    # not used in restore since base is present, but
                    # include values for human-readability
                    self._flatten_values_for_readability(obj, data)
            else:
                deepcopy_failed = True
        else:
            deepcopy_failed = True

        if deepcopy_failed:
            # store a deepcopy or fail
            if self.mode == "warn":
                msg = (
                    "ndarray is defined by reference to an object "
                    "we do not know how to serialize. "
                    "A deep copy is serialized instead, breaking "
                    "memory aliasing."
                )
                warnings.warn(msg)
            elif self.mode == "raise":
                msg = (
                    "ndarray is defined by reference to an object we do "
                    "not know how to serialize."
                )
                raise ValueError(msg)
            data = super().flatten(obj.copy(), data)

        return data

    def restore(self, data: dict[str, Any]) -> npt.NDArray[Any]:
        """decode numpy from json"""
        base = data.get("base", None)
        if base is None:
            # decode array with owndata=True
            arr = super().restore(data)
        else:
            # decode array view, which references the data of another array
            base = self.context.restore(base, reset=False)
            buffer: Any
            if isinstance(base, np.ndarray):
                buffer = base.data
                bufsize = base.nbytes
            elif isinstance(base, (bytes, bytearray, memoryview)):
                buffer = base
                bufsize = _buffer_nbytes(base)
            else:
                # the object is probably a nested list
                base = np.array(base)
                buffer = base.data
                bufsize = base.nbytes
            if isinstance(base, np.ndarray):
                assert base.flags.forc, (
                    "Current implementation assumes base is C or F contiguous"
                )

            dtype = self.restore_dtype(data).newbyteorder(data.get("byteorder", "|"))
            shape = data.get("shape")
            offset = data.get("offset", 0)
            strides = data.get("strides", None)
            if shape is not None:
                # never hand numpy a view that can reach outside its buffer
                _validate_buffer_range(bufsize, dtype, shape, offset, strides)

            arr = np.ndarray(
                buffer=buffer,
                dtype=dtype,
                shape=shape,  # type: ignore[arg-type]
                offset=offset,
                strides=strides,
            )

            self.restore_flags(data, arr)

        return arr


class NumpyUfuncHandler(BaseHandler):
    def flatten(self, obj: np.ufunc, data: dict[str, Any]) -> dict[str, Any]:
        data["importable_name"] = importable_name(obj)
        return data

    def restore(self, obj: dict[str, Any]) -> np.ufunc:
        # it seems proper practice to make mypy happy that it can't return None here
        # is to use the cast
        return cast(np.ufunc, loadclass(obj["importable_name"]))


def register_handlers(
    ndarray_mode: str = "warn",
    ndarray_size_threshold: int = 16,
    ndarray_compression: ModuleType = zlib,
) -> None:
    """Register handlers for numpy types

    :param ndarray_abc_xyz: Forward constructor arguments to NumpyNDArrayHandlerView.
        Options with an 'ndarray_' prefix correspond to the same-named
        NumpyNDArrayHandlerView constructor options, sans the 'ndarray_' prefix.
    """
    ndarray_handler = NumpyNDArrayHandlerView(
        mode=ndarray_mode,
        size_threshold=ndarray_size_threshold,
        compression=ndarray_compression,
    )
    register(np.ndarray, ndarray_handler, base=True)  # type: ignore[arg-type]
    register(np.dtype, NumpyDTypeHandler, base=True)
    register(np.generic, NumpyGenericHandler, base=True)
    # Numpy 1.20 has custom dtypes that must be registered separately.
    register(np.dtype(np.void).__class__, NumpyDTypeHandler, base=True)
    register(np.dtype(np.float32).__class__, NumpyDTypeHandler, base=True)
    register(np.dtype(np.int32).__class__, NumpyDTypeHandler, base=True)
    register(np.dtype(np.datetime64).__class__, NumpyDTypeHandler, base=True)
    register(np.datetime64, NumpyDatetimeHandler, base=True)
    # other handlers
    register(np.ufunc, NumpyUfuncHandler, base=True)


def unregister_handlers() -> None:
    """Remove numpy handlers from the handler registry"""
    unregister(np.dtype)
    unregister(np.generic)
    unregister(np.ndarray)
    # Numpy 1.20 dtypes
    unregister(np.dtype(np.void).__class__)
    unregister(np.dtype(np.float32).__class__)
    unregister(np.dtype(np.int32).__class__)
    unregister(np.dtype(np.datetime64).__class__)
    # other handlers
    unregister(np.ufunc)


def encode_oob(
    value: Any,
    context: "Pickler | None" = None,
    indent: "int | None" = None,
    separators: Any = None,
    **kwargs: Any,
) -> "tuple[str, list[bytes]]":
    """Encode value as JSON metadata plus an ordered out-of-band buffer table.

    This is the out-of-band (oob) counterpart of jsonpickle.encode for
    numpy arrays. Instead of inlining array payloads as base64 text
    (which roughly doubles the JSON size and memory footprint for large
    arrays), binary payloads are appended to an ordered buffer table and
    the JSON metadata only references buffers by index.

    :param value: the object to encode
    :param context: an optional pre-built jsonpickle.Pickler
    :param indent: pretty-printing indent, as in jsonpickle.encode
    :param separators: JSON separators, as in jsonpickle.encode
    :param kwargs: forwarded to jsonpickle.Pickler
    :return: a (metadata, buffers) pair where metadata is a JSON string
        and buffers is the ordered list of out-of-band buffers (bytes,
        compressed according to the registered ndarray handler's
        compression setting). Arrays below the handler's size_threshold
        are still inlined as text and produce no buffer entries.

    Views sharing one underlying allocation produce a single buffer
    entry; pass both return values to decode_oob to restore the object
    graph with sharing, dtypes, offsets and strides intact.
    """
    if context is None:
        context = Pickler(**kwargs)
    buffers: list[bytes] = []
    context.numpy_oob_buffers = buffers
    flattened = context.flatten(value, reset=True)
    payload = {OOB_TAG: {"value": flattened, "buffer_count": len(buffers)}}
    return json_backend.encode(payload, indent=indent, separators=separators), buffers


def decode_oob(
    metadata: str,
    buffers: Sequence[Any],
    context: "Unpickler | None" = None,
    classes: Any = None,
    **kwargs: Any,
) -> Any:
    """Decode JSON metadata produced by encode_oob.

    :param metadata: the JSON string returned by encode_oob
    :param buffers: the ordered buffer table returned by encode_oob.
        Each entry must be bytes, bytearray or a memoryview. Contiguous
        memoryviews are used zero-copy; non-contiguous memoryviews are
        explicitly copied into a packed C-contiguous bytes object. Any
        other type raises TypeError.
    :param context: an optional pre-built jsonpickle.Unpickler
    :param classes: forwarded to jsonpickle.Unpickler.restore
    :param kwargs: forwarded to jsonpickle.Unpickler

    The buffer count is checked against the metadata, every buffer
    reference is bounds-checked against the provided table, and every
    byte range reachable by a restored array or view is validated before
    it is handed to numpy; out-of-bounds views raise ValueError.
    """
    payload = json_backend.decode(metadata)
    if not (isinstance(payload, dict) and isinstance(payload.get(OOB_TAG), dict)):
        raise ValueError(
            "metadata was not produced by encode_oob(); "
            "use jsonpickle.decode() for regular jsonpickle output"
        )
    envelope = payload[OOB_TAG]
    expected = envelope.get("buffer_count")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        raise ValueError("corrupt out-of-band metadata: invalid buffer count")
    if len(buffers) != expected:
        raise ValueError(
            f"metadata expects {expected} out-of-band buffer(s), "
            f"but {len(buffers)} were provided"
        )
    normalized = [_normalize_oob_buffer(buf, i) for i, buf in enumerate(buffers)]
    is_ephemeral_context = context is None
    if context is None:
        context = Unpickler(**kwargs)
    context.numpy_oob_buffers = normalized
    result = context.restore(envelope["value"], reset=True, classes=classes)
    if is_ephemeral_context:
        # avoid holding references to external objects, mirroring
        # jsonpickle.decode
        context.reset()
    return result
