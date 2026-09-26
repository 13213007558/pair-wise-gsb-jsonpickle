import datetime
import json
import warnings
import zlib

import pytest

try:
    import numpy as np
    import numpy.testing as npt
    from numpy.testing import assert_equal
except ImportError:
    pytest.skip("numpy is not available", allow_module_level=True)

import jsonpickle
import jsonpickle.ext.numpy
from jsonpickle import handlers


@pytest.fixture(scope="module", autouse=True)
def numpy_extension():
    """Initialize the numpy extension for this test module"""
    jsonpickle.ext.numpy.register_handlers(ndarray_mode="warn")
    yield  # control to the test function.
    jsonpickle.ext.numpy.unregister_handlers()


def roundtrip(obj):
    return jsonpickle.decode(jsonpickle.encode(obj))


def test_dtype_roundtrip():
    dtypes = [
        np.int_,
        np.int32,
        np.float64,
        np.complex128,
        np.str_,
        np.object_,
        np.dtype(np.void),
        np.dtype(np.int32),
        np.dtype(np.float32),
        np.dtype("f4,i4,f2,i1"),
        np.dtype([("f0", "f4", (2,)), ("f1", "i4", (2,))]),
        np.dtype("i4", align=True),
        np.dtype("M8[7D]"),
        np.dtype(
            {
                "names": ["f0", "f1", "f2"],
                "formats": ["<u4", "<u2", "<u2"],
                "offsets": [0, 0, 2],
            },
            align=True,
        ),
        np.dtype([("f0", "i4"), ("f2", "i1")]),
        np.dtype(
            [
                (
                    "top",
                    [
                        ("tiles", (">f4", (64, 64)), (1,)),
                        ("rtile", ">f4", (64, 36)),
                    ],
                    (3,),
                ),
                (
                    "bottom",
                    [
                        ("bleft", (">f4", (8, 64)), (1,)),
                        ("bright", ">f4", (8, 36)),
                    ],
                ),
            ]
        ),
    ]

    for dtype in dtypes:
        encoded = jsonpickle.encode(dtype)
        decoded = jsonpickle.decode(encoded)
        assert dtype == decoded


def test_generic_roundtrip():
    values = [
        np.int_(1),
        np.int32(-2),
        np.float64(2.5),
        np.nan,
        -np.inf,
        np.inf,
        np.datetime64("2014-01-01"),
        np.str_("foo"),
        np.object_({"a": "b"}),
        np.complex128(1 - 2j),
    ]
    for value in values:
        decoded = roundtrip(value)
        assert_equal(decoded, value)
        assert isinstance(decoded, type(value))


def test_ndarray_roundtrip():
    arrays = [
        np.random.random((10, 20)),
        np.array([[True, False, True]]),
        np.array(["foo", "bar"]),
        np.array([b"baz"]),
        np.array(["2010", "NaT", "2030"]).astype("M"),
        np.rec.array(b"abcdefg" * 100, formats="i2,S3,i4", shape=3),
        np.rec.array(
            [
                (1, 11, "a"),
                (2, 22, "b"),
                (3, 33, "c"),
                (4, 44, "d"),
                (5, 55, "ex"),
                (6, 66, "f"),
                (7, 77, "g"),
            ],
            formats="u1,f4,S1",
        ),
        np.array(["1960-03-12", datetime.date(1960, 3, 12)], dtype="M8[D]"),
        np.array([0, 1, -1, np.inf, -np.inf, np.nan], dtype="f2"),
    ]

    arrays.extend(
        [
            np.rec.array(
                [("NGC1001", 11), ("NGC1002", 1.0), ("NGC1003", 1.0)],
                dtype=[("target", "S20"), ("V_mag", "f4")],
            )
        ]
    )
    for array in arrays:
        decoded = roundtrip(array)
        assert_equal(decoded, array)
        assert decoded.dtype == array.dtype


def test_shapes_containing_zeroes():
    """Test shapes which cannot be represented as nested lists"""
    expect = np.eye(3)[3:]
    actual = roundtrip(expect)
    npt.assert_array_equal(expect, actual)


def test_accuracy():
    """Test the accuracy of the string representation"""
    expect = np.random.randn(3, 3)
    actual = roundtrip(expect)
    npt.assert_array_equal(expect, actual)


def test_b64():
    """Test the binary encoding"""
    # Array of substantial size is stored as b64.
    expect = np.random.rand(10, 10)
    actual = roundtrip(expect)
    npt.assert_array_equal(expect, actual)


def test_views():
    """Test views under serialization"""
    rng = np.arange(20)  # a range of an array
    view = rng[10:]  # a view referencing a portion of an array
    data = [rng, view]

    actual = roundtrip(data)
    actual[0][15] = -1
    assert actual[1][5] == -1


def test_strides():
    """Test non-standard strides and offsets"""
    arr = np.eye(3)
    view = arr[1:, 1:]
    assert view.base is arr

    data = [arr, view]
    actual = roundtrip(data)

    # test that the deserialized arrays indeed view the same memory
    new_arr, new_view = actual
    new_arr[1, 2] = -1
    assert new_view[0, 1] == -1
    assert new_view.base is new_arr


def test_shared_base_after_referenced_object():
    """Regression test for #449.

    When an object flattened earlier registers a py/id reference that is
    not mirrored during restore (e.g. a masked array, whose decorative
    values were previously flattened through the pickler context), the id
    counters used by encode and decode drift apart. A later view stored as a
    py/id reference to a shared base then resolves to the wrong object and
    raises a strides-incompatible ValueError.
    """
    masked = np.ma.MaskedArray(np.array([1, 3], dtype=np.int64), mask=[False, False])
    base = np.arange(100, dtype=np.float64)
    # Two non-contiguous views sharing base. The second is stored as a
    # py/id reference to the base emitted by the first.
    view1 = base[0::50]
    view2 = base[1::50]
    assert view1.base is base and view2.base is base

    data = {"masked": masked, "view1": view1, "view2": view2}
    actual = roundtrip(data)

    npt.assert_array_equal(actual["masked"], masked)
    npt.assert_array_equal(actual["view1"], view1)
    npt.assert_array_equal(actual["view2"], view2)


def test_weird_arrays():
    """Test references to arrays that do not effectively own their memory"""
    base = np.arange(9)
    a = np.lib.stride_tricks.as_strided(base, shape=(9,), strides=(1,))
    b = a[5:]

    # this is kinda fishy; a has overlapping memory, _a does not
    warn_count = 1
    with warnings.catch_warnings(record=True) as w:
        _a = roundtrip(a)
        assert len(w) == warn_count
        npt.assert_array_equal(a, _a)

    # this also requires a deepcopy to work
    with warnings.catch_warnings(record=True) as w:
        _a, _b = roundtrip([a, b])
        assert len(w) == warn_count
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)


def test_transpose():
    """test handling of non-c-contiguous memory layout"""
    # simple case; view a c-contiguous array
    a = np.arange(9).reshape(3, 3)
    b = a[1:, 1:]
    assert b.base is a.base
    _a, _b = roundtrip([a, b])
    assert _b.base is _a.base
    npt.assert_array_equal(a, _a)
    npt.assert_array_equal(b, _b)

    # a and b both view the same contiguous array
    a = np.arange(9).reshape(3, 3).T
    b = a[1:, 1:]
    assert b.base is a.base
    _a, _b = roundtrip([a, b])
    assert _b.base is _a.base
    npt.assert_array_equal(a, _a)
    npt.assert_array_equal(b, _b)

    # view an f-contiguous array
    a = np.asfortranarray(np.arange(9).reshape(3, 3))
    b = a[1:, 1:]
    assert b.base is a
    _a, _b = roundtrip([a, b])
    assert _b.base is _a
    npt.assert_array_equal(a, _a)
    npt.assert_array_equal(b, _b)

    # now a.data.contiguous is False; we have to make a deepcopy to make
    # this work. note that this is a pretty contrived example though!
    base = np.arange(8).reshape(2, 2, 2).copy()
    a = np.lib.stride_tricks.as_strided(
        base,
        shape=base.shape,
        strides=(base.strides[0], base.strides[2], base.strides[1]),
    )
    b = a[1:, 1:]
    assert b.base is a

    warn_count = 1
    with warnings.catch_warnings(record=True) as w:
        _a, _b = roundtrip([a, b])
        assert len(w) == warn_count
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)


def test_fortran_base():
    """Test a base array in fortran order"""
    a = np.asfortranarray(np.arange(100).reshape((10, 10)))
    _a = roundtrip(a)
    npt.assert_array_equal(a, _a)


def test_buffer():
    """test behavior with memoryviews which are not ndarrays"""
    bstring = b"abcdefgh"
    a = np.frombuffer(bstring, dtype=np.byte)
    warn_count = 1
    with warnings.catch_warnings(record=True) as w:
        _a = roundtrip(a)
        npt.assert_array_equal(a, _a)
        assert len(w) == warn_count


def test_as_strided():
    """Test the result of as_strided()

    as_strided() returns an object that implements the array interface but
    is not an ndarray.

    """
    warn_count = 1
    a = np.arange(10)
    b = np.lib.stride_tricks.as_strided(a, shape=(5,), strides=(a.dtype.itemsize * 2,))
    data = [a, b]

    with warnings.catch_warnings(record=True) as w:
        # as_strided returns a DummyArray object, which we can not
        # currently serialize correctly FIXME: would be neat to add
        # support for all objects implementing the __array_interface__
        _data = roundtrip(data)
        assert len(w) == warn_count

    # as we were warned, deserialized result is no longer a view
    _data[0][0] = -1
    assert _data[1][0] == 0


def test_immutable():
    """test that immutability flag is copied correctly"""
    a = np.arange(10)
    a.flags.writeable = False
    _a = roundtrip(a)
    with pytest.raises(ValueError):
        _a[0] = 0


def test_byteorder():
    """Test the byteorder for text and binary encodings"""
    # small arr is stored as text
    a = np.arange(10)
    av = a.view(a.dtype.newbyteorder())
    b = a[:]
    bv = b.view(b.dtype.newbyteorder())
    _av, _bv = roundtrip([av, bv])
    npt.assert_array_equal(av, _av)
    npt.assert_array_equal(bv, _bv)

    # bigger arr is stored as binary
    a = np.arange(100)
    av = a.view(a.dtype.newbyteorder())
    b = a[:]
    bv = b.view(b.dtype.newbyteorder())
    _av, _bv = roundtrip([av, bv])
    npt.assert_array_equal(av, _av)
    npt.assert_array_equal(bv, _bv)


def test_zero_dimensional_array():
    expect = np.array(0.0, dtype="float64")
    actual = jsonpickle.decode(jsonpickle.encode(expect))
    npt.assert_array_equal(expect, actual)


def test_nested_data_list_of_dict_with_list_keys():
    """Ensure we can handle numpy arrays within a nested structure"""
    expect = [{"key": [np.array(0)]}]
    actual = roundtrip(expect)
    npt.assert_array_equal(expect[0]["key"][0], actual[0]["key"][0])

    expect = [{"key": [np.array([1.0])]}]
    actual = roundtrip(expect)
    npt.assert_array_equal(expect[0]["key"][0], actual[0]["key"][0])


def test_size_threshold_None():
    handler = jsonpickle.ext.numpy.NumpyNDArrayHandlerView(size_threshold=None)
    handlers.registry.unregister(np.ndarray)
    handlers.registry.register(np.ndarray, handler, base=True)
    expect = np.array([0, 1])
    actual = roundtrip(expect)
    npt.assert_array_equal(expect, actual)


def test_ndarray_dtype_object():
    a = np.array(["F" + str(i) for i in range(30)], dtype=object)
    buf = jsonpickle.encode(a)
    # This is critical for reproducing the numpy segfault issue when
    # restoring ndarray of dtype object.
    del a
    expect = np.array(["F" + str(i) for i in range(30)], dtype=object)
    actual = jsonpickle.decode(buf)
    npt.assert_array_equal(expect, actual)


def test_np_random():
    """Ensure random.random() arrays can be serialized"""
    obj = np.random.random(100)
    encoded = jsonpickle.encode(obj)
    clone = jsonpickle.decode(encoded)
    assert 100 == len(clone)
    for idx, (expect, actual) in enumerate(zip(obj, clone)):
        assert expect == actual


def test_np_poly1d():
    # issue 391, test poly1d roundtrip
    obj = np.poly1d([1, 2, 3])
    assert obj == jsonpickle.decode(jsonpickle.encode(obj))


TIME_UNITS = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")


def assert_scalar_roundtrip(obj):
    """A numpy scalar must come back with its dtype intact.

    Comparing datetime64/timedelta64 with == is not enough: numpy compares
    those across units, so a lost unit still compares equal.
    """
    decoded = roundtrip(obj)
    assert isinstance(decoded, np.generic), f"{type(decoded)} is not a scalar"
    assert decoded.dtype == obj.dtype, f"{obj.dtype} became {decoded.dtype}"
    assert_equal(decoded, obj)
    return decoded


def test_np_datetime64_units():
    """Ensure that we can roundtrip a numpy datetime64 with varying precisions"""
    for unit in TIME_UNITS:
        # int64 attoseconds only spans ~9.2s either side of the epoch, so the
        # stamp has to stay close to it to convert into ps/fs/as without
        # overflowing.
        stamp = np.datetime64("1970-01-01T00:00:01.789123456")
        assert_scalar_roundtrip(stamp.astype(f"datetime64[{unit}]"))
        assert_scalar_roundtrip(np.datetime64("NaT", unit))
    assert_scalar_roundtrip(np.datetime64("NaT"))


def test_np_datetime64_outside_datetime_range():
    """Dates outside the range of datetime.datetime flatten to a bare int,
    which cannot be restored without the declared unit"""
    # ns has ~584 years of int64 range, entirely inside datetime's, so no
    # date outside it can be expressed in ns.
    for unit in ("Y", "M", "W", "D", "h", "m", "s", "ms", "us"):
        assert_scalar_roundtrip(
            np.datetime64("30000-01-01").astype(f"datetime64[{unit}]")
        )


def test_np_timedelta64_units():
    """Ensure that we can roundtrip a numpy timedelta64 with varying precisions"""
    for unit in TIME_UNITS:
        assert_scalar_roundtrip(np.timedelta64(7, unit))
        assert_scalar_roundtrip(np.timedelta64(10**15, unit))
        assert_scalar_roundtrip(np.timedelta64("NaT", unit))
    assert_scalar_roundtrip(np.timedelta64(5))
    assert_scalar_roundtrip(np.timedelta64("NaT"))


def test_np_structured_scalar_roundtrip():
    """A structured scalar carries its field layout in the dtype, not the value"""
    dtype = [("a", "i4"), ("b", "f8"), ("c", "S3")]
    decoded = assert_scalar_roundtrip(np.array([(1, 2.5, b"xy")], dtype=dtype)[0])
    assert decoded["a"] == 1
    assert decoded["c"] == b"xy"


def test_np_extended_precision_scalar_roundtrip():
    """longdouble has no Python counterpart, so tolist() returns the scalar
    itself rather than a float"""
    assert_scalar_roundtrip(np.longdouble("1.1"))
    assert_scalar_roundtrip(np.clongdouble(complex(1.1, 2.0)))


def test_np_ufuncs():
    obj = np.log1p
    encoded = jsonpickle.encode(obj)
    assert obj is jsonpickle.decode(encoded)


class ThingWithDel:
    def __init__(self, data):
        self.data = data
        self.fn = None

    def __del__(self):
        if self.fn is not None:
            self.fn()


def test_no_external_references():
    """Ensure that no external references are held"""
    state = {"count": 0}  # How many times has the destructor been called?

    def destructor():
        state["count"] += 1

    obj = ThingWithDel(data=np.array([1, 2, 3]))

    obj_encoded = jsonpickle.encode(obj)
    obj.fn = destructor
    del obj

    obj2 = jsonpickle.decode(obj_encoded)
    obj2.fn = destructor
    del obj2

    assert state["count"] == 2


def oob_roundtrip(obj):
    metadata, buffers = jsonpickle.ext.numpy.encode_oob(obj)
    return jsonpickle.ext.numpy.decode_oob(metadata, buffers)


@pytest.fixture
def default_ndarray_handler():
    """Re-register the default ndarray handler.

    test_size_threshold_None replaces the global ndarray handler with a
    text-only one; the oob tests need the default binary configuration.
    """
    jsonpickle.ext.numpy.register_handlers()
    yield


def test_oob_roundtrip_matches_default_mode(default_ndarray_handler):
    """Out-of-band buffers restore the same values as the base64 default"""
    arr = np.random.rand(10, 10)
    metadata, buffers = jsonpickle.ext.numpy.encode_oob(arr)
    assert isinstance(metadata, str)
    assert len(buffers) == 1
    assert all(isinstance(buf, bytes) for buf in buffers)
    # the buffer holds exactly what the default mode would base64-encode
    assert zlib.decompress(buffers[0]) == arr.tobytes(order="A")

    actual = jsonpickle.ext.numpy.decode_oob(metadata, buffers)
    npt.assert_array_equal(arr, actual)
    npt.assert_array_equal(roundtrip(arr), actual)


def test_oob_small_array_stays_inline(default_ndarray_handler):
    """Arrays below the size threshold produce no buffer entries"""
    arr = np.arange(3)
    metadata, buffers = jsonpickle.ext.numpy.encode_oob(arr)
    assert buffers == []
    npt.assert_array_equal(jsonpickle.ext.numpy.decode_oob(metadata, []), arr)


def test_oob_shared_base_single_buffer(default_ndarray_handler):
    """Views of one allocation share a single out-of-band buffer"""
    rng = np.arange(20)
    view = rng[10:]
    metadata, buffers = jsonpickle.ext.numpy.encode_oob([rng, view])
    assert len(buffers) == 1

    new_rng, new_view = jsonpickle.ext.numpy.decode_oob(metadata, buffers)
    assert new_view.base is new_rng
    new_rng[15] = -1
    assert new_view[5] == -1


def test_oob_overlapping_views(default_ndarray_handler):
    """Mutations in the overlap of two views are visible from both sides"""
    base = np.arange(100)
    view1 = base[20:80]
    view2 = base[40:100]
    metadata, buffers = jsonpickle.ext.numpy.encode_oob([view1, view2])
    assert len(buffers) == 1

    new_view1, new_view2 = jsonpickle.ext.numpy.decode_oob(metadata, buffers)
    npt.assert_array_equal(new_view1, view1)
    npt.assert_array_equal(new_view2, view2)
    new_view1[30] = -1  # base[50], inside the overlap
    assert new_view2[10] == -1


def test_oob_strides_and_offset(default_ndarray_handler):
    """Positive and negative strides plus offsets survive the roundtrip"""
    base = np.arange(100)
    shifted = base[30:]
    reversed_view = base[::-1]
    assert reversed_view.strides[0] < 0

    metadata, buffers = jsonpickle.ext.numpy.encode_oob([base, shifted, reversed_view])
    assert len(buffers) == 1
    new_base, new_shifted, new_reversed = jsonpickle.ext.numpy.decode_oob(
        metadata, buffers
    )

    npt.assert_array_equal(new_shifted, shifted)
    npt.assert_array_equal(new_reversed, reversed_view)
    assert new_reversed.strides == reversed_view.strides
    assert new_shifted.base is new_base
    assert new_reversed.base is new_base
    new_base[30] = -1
    assert new_shifted[0] == -1
    assert new_reversed[69] == -1


def test_oob_dtype_object_roundtrip(default_ndarray_handler):
    """Object arrays take the binary buffer path without segfaulting"""
    arr = np.array([str(i) for i in range(30)], dtype=object)
    metadata, buffers = jsonpickle.ext.numpy.encode_oob(arr)
    assert len(buffers) == 1
    actual = jsonpickle.ext.numpy.decode_oob(metadata, buffers)
    assert list(actual) == list(arr)


def test_oob_buffer_count_mismatch(default_ndarray_handler):
    """The provided buffer table must match the metadata"""
    metadata, buffers = jsonpickle.ext.numpy.encode_oob(np.arange(100))
    with pytest.raises(ValueError):
        jsonpickle.ext.numpy.decode_oob(metadata, [])
    with pytest.raises(ValueError):
        jsonpickle.ext.numpy.decode_oob(metadata, buffers + [b"extra"])


def test_oob_rejects_regular_metadata(default_ndarray_handler):
    """decode_oob refuses metadata that was not produced by encode_oob"""
    with pytest.raises(ValueError):
        jsonpickle.ext.numpy.decode_oob(jsonpickle.encode(np.arange(100)), [])


def test_oob_metadata_rejected_by_default_decode(default_ndarray_handler):
    """Regular decode() points oob metadata back at decode_oob"""
    metadata, _buffers = jsonpickle.ext.numpy.encode_oob(np.arange(100))
    with pytest.raises(ValueError):
        jsonpickle.decode(metadata)


def test_oob_truncated_buffer(default_ndarray_handler):
    """A short buffer is rejected before numpy sees it"""
    metadata, buffers = jsonpickle.ext.numpy.encode_oob(np.arange(100))
    buffers[0] = zlib.compress(zlib.decompress(buffers[0])[:-1])
    with pytest.raises(ValueError):
        jsonpickle.ext.numpy.decode_oob(metadata, buffers)


def _find_first(node, key):
    """Return the first dict in a decoded-json tree that contains key"""
    if isinstance(node, dict):
        if key in node:
            return node
        for value in node.values():
            found = _find_first(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_first(value, key)
            if found is not None:
                return found
    return None


def test_oob_out_of_bounds_view_rejected(default_ndarray_handler):
    """A tampered view offset is rejected before numpy sees it"""
    base = np.arange(100)
    view = base[5:]
    metadata, buffers = jsonpickle.ext.numpy.encode_oob([base, view])

    payload = json.loads(metadata)
    _find_first(payload, "offset")["offset"] = 10**9
    with pytest.raises(ValueError):
        jsonpickle.ext.numpy.decode_oob(json.dumps(payload), buffers)


def test_oob_out_of_range_buffer_index_rejected(default_ndarray_handler):
    """A tampered buffer index is rejected"""
    metadata, buffers = jsonpickle.ext.numpy.encode_oob(np.arange(100))
    payload = json.loads(metadata)
    tag = jsonpickle.ext.numpy.OOB_BUFFER_TAG
    _find_first(payload, tag)[tag] = 7
    with pytest.raises(ValueError):
        jsonpickle.ext.numpy.decode_oob(json.dumps(payload), buffers)


def test_oob_buffer_types(default_ndarray_handler):
    """Buffers may be bytes, bytearray or memoryview"""
    arr = np.arange(100)
    metadata, buffers = jsonpickle.ext.numpy.encode_oob(arr)
    raw = buffers[0]

    actual = jsonpickle.ext.numpy.decode_oob(metadata, [bytearray(raw)])
    npt.assert_array_equal(arr, actual)

    actual = jsonpickle.ext.numpy.decode_oob(metadata, [memoryview(raw)])
    npt.assert_array_equal(arr, actual)

    # a non-contiguous memoryview is explicitly copied into packed bytes
    interleaved = bytearray()
    for byte in raw:
        interleaved += bytes([byte, 0])
    strided_view = memoryview(interleaved)[::2]
    assert not strided_view.c_contiguous
    actual = jsonpickle.ext.numpy.decode_oob(metadata, [strided_view])
    npt.assert_array_equal(arr, actual)

    with pytest.raises(TypeError):
        jsonpickle.ext.numpy.decode_oob(metadata, ["not a buffer"])


def test_oob_default_api_unchanged(default_ndarray_handler):
    """encode/decode keep their signatures and return types"""
    arr = np.arange(100)
    encoded = jsonpickle.encode(arr)
    assert isinstance(encoded, str)
    npt.assert_array_equal(jsonpickle.decode(encoded), arr)
