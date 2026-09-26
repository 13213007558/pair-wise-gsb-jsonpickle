import gc
import weakref

import pytest

import jsonpickle


class Thing:
    def __init__(self, name):
        self.name = name


class SelfReferential:
    def __init__(self):
        self.self_ref = weakref.ref(self)


def roundtrip(obj):
    return jsonpickle.decode(jsonpickle.encode(obj))


def test_live_weakref_roundtrip():
    target = Thing("a")
    restored = roundtrip({"strong": target, "weak": weakref.ref(target)})
    assert restored["weak"]() is restored["strong"]
    assert restored["weak"]().name == "a"


def test_weakref_alone_does_not_snapshot_target():
    target = Thing("a")
    encoded = jsonpickle.encode({"weak": weakref.ref(target)})
    assert "py/object" not in encoded
    restored = jsonpickle.decode(encoded)
    assert isinstance(restored["weak"], weakref.ReferenceType)
    assert restored["weak"]() is None


def test_dead_weakref_stays_dead():
    target = Thing("a")
    ref = weakref.ref(target)
    del target
    gc.collect()
    assert ref() is None
    restored = roundtrip({"weak": ref})
    assert isinstance(restored["weak"], weakref.ReferenceType)
    assert restored["weak"]() is None


def test_weakref_not_prematurely_dead_after_decode():
    target = Thing("a")
    encoded = jsonpickle.encode({"strong": target, "weak": weakref.ref(target)})
    del target
    gc.collect()
    restored = jsonpickle.decode(encoded)
    # the strong edge in the snapshot keeps the referent alive
    gc.collect()
    assert restored["weak"]() is restored["strong"]


def test_weakref_dies_once_strong_refs_are_dropped():
    target = Thing("a")
    encoded = jsonpickle.encode({"strong": target, "weak": weakref.ref(target)})
    del target
    restored = jsonpickle.decode(encoded)
    ref = restored["weak"]
    assert ref() is not None
    del restored
    gc.collect()
    # neither the unpickler's reference table nor any module-level
    # cache may keep the referent alive
    assert ref() is None


def test_multiple_weakrefs_to_same_target():
    target = Thing("a")
    restored = roundtrip(
        {"strong": target, "weak1": weakref.ref(target), "weak2": weakref.ref(target)}
    )
    assert restored["weak1"]() is restored["strong"]
    assert restored["weak2"]() is restored["strong"]
    assert restored["weak1"]() is restored["weak2"]()


def test_weakref_to_target_later_in_snapshot():
    target = Thing("a")
    restored = roundtrip([weakref.ref(target), target])
    assert restored[0]() is restored[1]


def test_weakref_cycle():
    restored = roundtrip(SelfReferential())
    assert restored.self_ref() is restored


def test_weakref_to_object_in_cycle():
    first = Thing("first")
    second = Thing("second")
    first.other = second
    second.other = first
    second.weak = weakref.ref(first)
    restored = roundtrip(second)
    assert restored.weak() is restored.other


def test_weakref_as_instance_attribute():
    target = Thing("a")
    holder = Thing("holder")
    holder.ref = weakref.ref(target)
    restored = roundtrip({"holder": holder, "target": target})
    assert restored["holder"].ref() is restored["target"]


def test_weakref_with_callback_is_rejected():
    target = Thing("a")
    ref = weakref.ref(target, lambda ref: None)
    with pytest.raises(ValueError, match="callback"):
        jsonpickle.encode({"weak": ref, "strong": target})


def test_weakref_unpicklable_mode():
    target = Thing("a")
    encoded = jsonpickle.encode(
        {"strong": target, "weak": weakref.ref(target)}, unpicklable=False
    )
    assert jsonpickle.decode(encoded)["weak"] is None
