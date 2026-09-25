import gc
import weakref

import pytest

import jsonpickle


class Weakrefable:
    def __init__(self):
        self.ref = None
        self.peer = None


def test_live_weakref_roundtrip():
    target = Weakrefable()
    payload = jsonpickle.encode({"target": target, "ref": weakref.ref(target)})
    result = jsonpickle.decode(payload)
    ref = result["ref"]
    assert isinstance(ref, weakref.ReferenceType)
    assert ref() is result["target"]


def test_dead_weakref_restores_dead():
    ref = weakref.ref(Weakrefable())
    assert ref() is None
    restored = jsonpickle.decode(jsonpickle.encode([ref]))
    assert isinstance(restored[0], weakref.ReferenceType)
    assert restored[0]() is None


def test_weak_edge_does_not_capture_target():
    target = Weakrefable()
    ref = weakref.ref(target)
    # The target is alive but only reachable through the weak edge, so it
    # must not be pulled into the snapshot.
    payload = jsonpickle.encode([ref])
    del target
    gc.collect()
    restored = jsonpickle.decode(payload)
    assert restored[0]() is None


def test_weakref_dies_once_strong_refs_are_gone():
    target = Weakrefable()
    payload = jsonpickle.encode({"target": target, "ref": weakref.ref(target)})
    del target
    gc.collect()
    result = jsonpickle.decode(payload)
    ref = result["ref"]
    # No premature death right after restoration.
    assert ref() is result["target"]
    # Dropping the last strong references must kill the weakref; neither the
    # decoder's reference table nor any module-level cache may keep it alive.
    del result
    gc.collect()
    assert ref() is None


def test_pickler_does_not_keep_target_alive():
    target = Weakrefable()
    ref = weakref.ref(target)
    jsonpickle.encode({"target": target, "ref": ref})
    del target
    gc.collect()
    assert ref() is None


def test_shared_weakref_identity_preserved():
    target = Weakrefable()
    ref = weakref.ref(target)
    restored = jsonpickle.decode(jsonpickle.encode([ref, ref, target]))
    assert restored[0] is restored[1]
    assert restored[0]() is restored[2]
    assert restored[1]() is restored[2]


def test_weakref_to_later_target():
    # The target shows up after the weakref in serialization order.
    target = Weakrefable()
    restored = jsonpickle.decode(jsonpickle.encode([weakref.ref(target), target]))
    assert restored[0]() is restored[1]


def test_weakref_cycle():
    node = Weakrefable()
    node.ref = weakref.ref(node)
    restored = jsonpickle.decode(jsonpickle.encode(node))
    assert restored.ref() is restored


def test_weakref_mutual_cycle():
    first = Weakrefable()
    second = Weakrefable()
    first.ref = weakref.ref(second)
    second.peer = first
    restored = jsonpickle.decode(jsonpickle.encode({"a": first, "b": second}))
    assert restored["a"].ref() is restored["b"]
    assert restored["b"].peer is restored["a"]


def test_weakref_with_callback_rejected():
    target = Weakrefable()
    ref = weakref.ref(target, lambda r: None)
    with pytest.raises(ValueError, match="callback"):
        jsonpickle.encode({"target": target, "ref": ref})


def test_weakref_unpicklable_mode_encodes_dead():
    target = Weakrefable()
    ref = weakref.ref(target)
    payload = jsonpickle.encode([ref], unpicklable=False)
    assert jsonpickle.decode(payload)[0]() is None
