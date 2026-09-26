from __future__ import annotations

import ast
import json
import sys
import warnings
import zlib
from collections.abc import Iterable
from types import ModuleType
from typing import Any, NoReturn, Protocol, cast

import numpy as np
import numpy.typing as npt

from ..handlers import BaseHandler, register, unregister
from ..pickler import Pickler
from ..pickler import encode as _encode_json
from ..unpickler import Unpickler
from ..unpickler import decode as _decode_json
from ..util import b64decode, b64encode, importable_name, loadclass

__all__ = [
    "BUFFER_TAG",
    "decode",
    "encode",
    "register_handlers",
    "unregister_handlers",
]


# matches `np.ndarray`, `np.generic`, and anything else with a `.dtype` attribute
class _HasDType(Protocol):
    @property
    def dtype(self) -> np.dtype[Any]: ...


native_byteorder: str = "<" if sys.byteorder == "little" else ">"


def get_byteorder(arr: _HasDType) -> str:
    """translate equals sign to native order"""
    byteorder = arr.dtype.byteorder
    return native_byteorder if byteorder == "=" else byteorder


# Tag used in the JSON metadata to reference an out-of-band buffer by its
# index in the buffer table returned by encode()
BUFFER_TAG = "py/buffer"

# Constructor options forwarded when encode()/decode() build their own
# pickling context; anything else requires an explicit context= argument
_PICKLER_OPTION_NAMES = (
    "unpicklable",
    "make_refs",
    "keys",
    "max_depth",
    "warn",
    "use_base85",
    "fail_safe",
    "include_properties",
    "handle_readonly",
    "handler_context",
)
_UNPICKLER_OPTION_NAMES = (
    "keys",
    "safe",
    "on_missing",
    "handle_readonly",
    "handler_context",
)


def _oob_buffers(context: Any) -> list[bytes] | None:
    """Return the out-of-band buffer table of a pickler context.

    Returns None when the context is not encoding in out-of-band mode.
    """
    return getattr(context, "numpy_oob_buffers", None)


def _stash_buffer(
    context: Any, buf: bytes, key: tuple[int, int] | None = None
) -> int:
    """Append buf to the context's out-of-band buffer table.

    When key (a memory (address, length) pair) is given, buffers are
    deduplicated so that one underlying allocation appears exactly once,
    no matter how many views reference it.
    """
    buffers: list[bytes] = context.numpy_oob_buffers
    dedup: dict[tuple[int, int], int] | None = None
    if key is not None:
        dedup = context.__dict__.setdefault("_numpy_oob_dedup", {})
        index = dedup.get(key)
        if index is not None:
            return index
    index = len(buffers)
    buffers.append(bytes(buf))
    if dedup is not None:
        assert key is not None
        dedup[key] = index
    return index


def _buffer_index(value: Any) -> Any:
    """Return the index if value is a buffer reference, else None"""
    if isinstance(value, dict) and BUFFER_TAG in value:
        return value[BUFFER_TAG]
    return None


def _normalize_provided_buffer(index: int, buffer: Any) -> npt.NDArray[Any]:
    """Validate a caller-provided out-of-band buffer as a flat uint8 array.

    bytes, bytearray and contiguous memoryview objects are accepted
    directly. Non-contiguous memoryviews are explicitly copied via
    memoryview.tobytes(); anything else is rejected with a TypeError.
    """
    if isinstance(buffer, memoryview) and not buffer.contiguous:
        # explicit copy: a non-contiguous view cannot back an ndarray
        buffer = buffer.tobytes()
    try:
        return np.frombuffer(buffer, dtype=np.uint8)
    except (TypeError, ValueError, BufferError) as error:
        raise TypeError(
            f"out-of-band buffer {index} must be a contiguous bytes-like "
            "object (bytes, bytearray, or contiguous memoryview), got "
            f"{type(buffer).__name__}"
        ) from error


def _oob_array(
    context: Any, index: Any, compression: ModuleType | None
) -> npt.NDArray[Any]:
    """Return the shared writable uint8 array for out-of-band buffer index.

    The array is cached on the unpickler context, so every array restored
    from the same buffer index shares one memory allocation. Writable
    caller-provided buffers are used in place (zero-copy); read-only
    buffers are copied exactly once so that restored arrays are writeable,
    matching the default mode.
    """
    prepared = getattr(context, "numpy_oob_buffers", None)
    if prepared is None:
        raise ValueError(
            f"snapshot references out-of-band buffer {index!r}; decode it "
            "with jsonpickle.ext.numpy.decode() and supply the buffer table"
        )
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(prepared)
    ):
        raise ValueError(
            f"out-of-band buffer index {index!r} is out of range; "
            f"{len(prepared)} buffer(s) were provided"
        )
    cache = context.__dict__.setdefault("_numpy_oob_cache", {})
    arr = cache.get(index)
    if arr is None:
        raw = prepared[index]
        if compression is None:
            arr = raw
        else:
            arr = np.frombuffer(compression.decompress(raw.data), dtype=np.uint8)
        if not arr.flags.writeable:
            arr = arr.copy()
        cache[index] = arr
    return arr


def _buffer_nbytes(buffer: Any) -> int:
    if isinstance(buffer, (np.ndarray, memoryview)):
        return buffer.nbytes  # type: ignore[return-value]
    return len(buffer)


def _check_view_bounds(
    nbytes: int,
    offset: Any,
    shape: tuple[int, ...],
    strides: Any,
    itemsize: int,
) -> None:
    """Ensure all bytes reachable through (offset, shape, strides) lie
    inside a buffer of nbytes bytes.

    NumPy does not validate the downward reach of negative strides, so an
    out-of-bounds view must be rejected here rather than handed to
    np.ndarray.
    """
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError(f"array view offset {offset!r} is not a non-negative int")
    if offset > nbytes:
        raise ValueError(
            f"array view offset {offset} exceeds buffer of {nbytes} bytes"
        )
    if strides is not None:
        strides = tuple(strides)
        if len(strides) != len(shape):
            raise ValueError(
                f"array view has {len(shape)} dimensions but "
                f"{len(strides)} strides"
            )
    if 0 in shape:
        return  # no bytes are reachable
    low = high = offset
    if strides is None:
        span = itemsize
        for dim in shape:
            span *= dim
        high = offset + span
    else:
        for dim, stride in zip(shape, strides):
            step = (dim - 1) * stride
            if step < 0:
                low += step
            else:
                high += step
        high += itemsize
    if low < 0 or high > nbytes:
        raise ValueError(
            f"array view reaches byte range [{low}, {high}), outside the "
            f"buffer of {nbytes} bytes"
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
            buffers = _oob_buffers(self.context)
            if buffers is None:
                data["values"] = b64encode(buf)
            else:
                # out-of-band mode: reference the buffer table by index
                data["values"] = {BUFFER_TAG: _stash_buffer(self.context, buf)}
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
            buffer_index = _buffer_index(values)
            if buffer_index is not None:
                # out-of-band mode: the payload lives in the
                # caller-provided buffer table
                buf = _oob_array(self.context, buffer_index, self.compression)
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
            elif buffer_index is not None:
                shape = tuple(data.get("shape") or ())
                _check_view_bounds(buf.nbytes, 0, shape, None, dtype.itemsize)
                arr = np.ndarray(
                    buffer=buf,
                    dtype=dtype,
                    shape=shape,
                    order=data.get("order", "C"),
                )
                self.restore_byteorder(data, arr)
            else:
                arr = np.ndarray(
                    buffer=buf,
                    dtype=dtype,
                    shape=data.get("shape"),  # type: ignore[arg-type]
                    order=data.get("order", "C"),
                ).copy()  # make a copy, to force the result to own the data
                self.restore_byteorder(data, arr)
            self.restore_flags(data, arr)

        return arr


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

                buffers = _oob_buffers(self.context)
                if buffers is None:
                    data["base"] = self.context.flatten(base_store, reset=False)
                else:
                    # out-of-band mode: stash the buffer bytes and reference
                    # them by index; deduplicate by allocation so that one
                    # underlying buffer appears exactly once, no matter how
                    # many views of it are serialized
                    stored = bytes(base_store)
                    if self.compression:
                        stored = self.compression.compress(stored)
                    key = (base_buf.ctypes.data, base_buf.nbytes)
                    data["base"] = {
                        BUFFER_TAG: _stash_buffer(self.context, stored, key=key)
                    }

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
            buffer_index = _buffer_index(base)
            buffer: Any
            if buffer_index is not None:
                # out-of-band mode: all views of one buffer index share
                # a single allocation
                base = None
                buffer = _oob_array(self.context, buffer_index, self.compression)
            else:
                base = self.context.restore(base, reset=False)
                if isinstance(base, np.ndarray):
                    buffer = base.data
                elif isinstance(base, (bytes, bytearray, memoryview)):
                    buffer = base
                else:
                    # the object is probably a nested list
                    base = np.array(base)
                    buffer = base.data
                if isinstance(base, np.ndarray):
                    assert base.flags.forc, (
                        "Current implementation assumes base is C or F contiguous"
                    )

            shape = data.get("shape")
            offset = data.get("offset", 0)
            strides = data.get("strides", None)
            dtype = self.restore_dtype(data).newbyteorder(data.get("byteorder", "|"))
            _check_view_bounds(
                _buffer_nbytes(buffer),
                offset,
                tuple(shape) if shape is not None else (),
                strides,
                dtype.itemsize,
            )
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


def encode(value: Any, **kwargs: Any) -> tuple[str, list[bytes]]:
    """Encode value to JSON with numpy array payloads out-of-band.

    Returns a (json_text, buffers) tuple. json_text is the JSON metadata
    in which every binary array payload is replaced by a
    {"py/buffer": index} reference, and buffers is the ordered buffer
    table those indexes refer to. Views of one underlying allocation
    share a single buffer entry. Use decode() with the same buffer table
    to restore the value.

    Keyword arguments are forwarded to jsonpickle.encode(); pass a
    pre-configured jsonpickle.Pickler as context= for full control over
    pickler options. The plain jsonpickle.encode() is unaffected and
    keeps embedding payloads as base64 text.
    """
    context = kwargs.get("context")
    if context is None:
        options = {k: kwargs[k] for k in _PICKLER_OPTION_NAMES if k in kwargs}
        context = Pickler(**options)
        kwargs["context"] = context
    buffers: list[bytes] = []
    context.numpy_oob_buffers = buffers
    return _encode_json(value, **kwargs), buffers


def decode(json_text: str, buffers: Iterable[Any], **kwargs: Any) -> Any:
    """Restore a value encoded with encode().

    buffers is the ordered buffer table returned by encode(). Each entry
    must be bytes, bytearray or a contiguous memoryview; non-contiguous
    memoryviews are explicitly copied via memoryview.tobytes(). Writable
    buffers are used in place (zero-copy: arrays restored from them share
    memory with the caller's buffer); read-only buffers are copied once
    so that restored arrays are writeable, matching the default mode.

    The buffer count, buffer lengths, and every byte range reachable
    through the shape/strides/offset metadata are validated before
    anything is handed to NumPy; out-of-bounds views raise ValueError.

    Keyword arguments are forwarded to jsonpickle.decode(); pass a
    pre-configured jsonpickle.Unpickler as context= for full control
    over unpickler options.
    """
    context = kwargs.get("context")
    if context is None:
        options = {k: kwargs[k] for k in _UNPICKLER_OPTION_NAMES if k in kwargs}
        context = Unpickler(**options)
        kwargs["context"] = context
    context.numpy_oob_buffers = [
        _normalize_provided_buffer(index, buffer)
        for index, buffer in enumerate(buffers)
    ]
    return _decode_json(json_text, **kwargs)


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
