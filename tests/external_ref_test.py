import pytest

import jsonpickle
from jsonpickle import errors, tags
from jsonpickle.pickler import Pickler
from jsonpickle.unpickler import Unpickler


class Account:
    def __init__(self, name, settings):
        self.name = name
        self.settings = settings


class Node:
    def __init__(self, name):
        self.name = name
        self.peer = None
        self.config = None


class CountingResolver:
    def __init__(self, objects, expected_type=None):
        self.objects = objects
        self.calls = []
        if expected_type is not None:
            self.expected_type = expected_type

    def resolve(self, identifier):
        self.calls.append(identifier)
        return self.objects[identifier]


def make_shared_settings():
    return {"theme": "dark", "limits": {"cpu": 4, "memory": "16Gi"}}


def encode_with_external(value, selected, identifier="cfg://shared"):
    return jsonpickle.encode(
        value, external_ref=lambda obj: identifier if obj is selected else None
    )


def test_shared_external_dict_acceptance():
    """Two business objects share one external dict: the snapshot omits the
    dict contents, both references resolve to a single instance, and a
    mutation through one reference is visible through the other."""
    shared = make_shared_settings()
    accounts = [Account("alice", shared), Account("bob", shared)]

    payload = encode_with_external(accounts, shared)

    # the snapshot contains a reference, not the dict contents
    assert "dark" not in payload
    assert "16Gi" not in payload
    assert tags.EXTERNAL in payload

    resolver = CountingResolver({"cfg://shared": shared})
    restored = jsonpickle.decode(payload, external_resolver=resolver)

    alice, bob = restored
    assert alice.name == "alice"
    assert bob.name == "bob"
    # the identifier was resolved exactly once and both references
    # received the same instance
    assert resolver.calls == ["cfg://shared"]
    assert alice.settings is bob.settings
    assert alice.settings is shared

    alice.settings["theme"] = "light"
    assert bob.settings["theme"] == "light"


def test_external_object_is_not_expanded_at_root():
    shared = make_shared_settings()
    payload = encode_with_external(shared, shared)
    assert (
        jsonpickle.decode(payload, external_resolver={"cfg://shared": shared}) is shared
    )


def test_unselected_objects_use_regular_protocol():
    shared = make_shared_settings()
    local = {"note": "inline"}
    account = Account("alice", shared)
    account.local = local

    payload = encode_with_external(account, shared)
    assert "inline" in payload
    assert "dark" not in payload

    restored = jsonpickle.decode(payload, external_resolver={"cfg://shared": shared})
    assert restored.settings is shared
    assert restored.local == local
    # the inline dict went through the regular protocol and was copied
    assert restored.local is not local


def test_cycles_in_regular_graph_still_work():
    shared = make_shared_settings()
    first = Node("first")
    second = Node("second")
    first.peer = second
    second.peer = first
    first.config = shared
    second.config = shared

    payload = encode_with_external(first, shared)
    resolver = CountingResolver({"cfg://shared": shared})
    restored = jsonpickle.decode(payload, external_resolver=resolver)

    assert restored.peer.peer is restored
    assert restored.config is restored.peer.config is shared
    assert resolver.calls == ["cfg://shared"]


def test_resolver_mapping_and_callable_forms():
    shared = make_shared_settings()
    payload = encode_with_external(Account("alice", shared), shared)

    as_mapping = jsonpickle.decode(payload, external_resolver={"cfg://shared": shared})
    assert as_mapping.settings is shared

    as_callable = jsonpickle.decode(
        payload, external_resolver=lambda identifier: shared
    )
    assert as_callable.settings is shared


def test_cache_is_not_shared_across_independent_decodes():
    shared = make_shared_settings()
    accounts = [Account("alice", shared), Account("bob", shared)]
    payload = encode_with_external(accounts, shared)

    def factory(identifier):
        return make_shared_settings()

    first = jsonpickle.decode(payload, external_resolver=factory)
    second = jsonpickle.decode(payload, external_resolver=factory)

    # within one decode all references share a single instance
    assert first[0].settings is first[1].settings
    assert second[0].settings is second[1].settings
    # independent decode calls do not share the resolution cache
    assert first[0].settings is not second[0].settings


def test_explicit_context_objects():
    shared = make_shared_settings()
    account = Account("alice", shared)

    pickler = Pickler(
        external_ref=lambda obj: "cfg://shared" if obj is shared else None
    )
    payload = jsonpickle.encode(account, context=pickler)
    assert "dark" not in payload

    unpickler = Unpickler(external_resolver={"cfg://shared": shared})
    restored = jsonpickle.decode(payload, context=unpickler)
    assert restored.settings is shared


def test_kwargs_override_explicit_context():
    shared = make_shared_settings()
    account = Account("alice", shared)

    payload = jsonpickle.encode(
        account,
        context=Pickler(),
        external_ref=lambda obj: "cfg://shared" if obj is shared else None,
    )
    assert "dark" not in payload

    restored = jsonpickle.decode(
        payload, context=Unpickler(), external_resolver={"cfg://shared": shared}
    )
    assert restored.settings is shared


def test_unknown_identifier_raises_not_found():
    shared = make_shared_settings()
    payload = encode_with_external(Account("alice", shared), shared)

    with pytest.raises(errors.ExternalReferenceNotFoundError):
        jsonpickle.decode(payload, external_resolver={})

    def raising_resolver(identifier):
        raise KeyError(identifier)

    with pytest.raises(errors.ExternalReferenceNotFoundError):
        jsonpickle.decode(payload, external_resolver=raising_resolver)


def test_resolver_failure_raises_resolver_error():
    shared = make_shared_settings()
    payload = encode_with_external(Account("alice", shared), shared)

    def broken_resolver(identifier):
        raise RuntimeError("repository unavailable")

    with pytest.raises(errors.ExternalResolverError):
        jsonpickle.decode(payload, external_resolver=broken_resolver)


def test_missing_resolver_raises_instead_of_falling_back():
    shared = make_shared_settings()
    payload = encode_with_external(Account("alice", shared), shared)

    with pytest.raises(errors.ExternalResolverError):
        jsonpickle.decode(payload)


def test_type_mismatch_raises_type_error():
    shared = make_shared_settings()
    payload = encode_with_external(Account("alice", shared), shared)

    resolver = CountingResolver({"cfg://shared": shared}, expected_type=list)
    with pytest.raises(errors.ExternalReferenceTypeError):
        jsonpickle.decode(payload, external_resolver=resolver)

    # a conforming declaration resolves normally
    resolver = CountingResolver({"cfg://shared": shared}, expected_type=dict)
    restored = jsonpickle.decode(payload, external_resolver=resolver)
    assert restored.settings is shared


def test_literal_external_key_roundtrips():
    """A user dict containing the reserved wire key is escaped, not
    interpreted as an external reference."""
    data = {tags.EXTERNAL: "not-a-reference"}
    payload = jsonpickle.encode(data)
    assert jsonpickle.decode(payload) == data
