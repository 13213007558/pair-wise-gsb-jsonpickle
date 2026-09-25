"""Tests for the py/external external reference protocol."""

import pytest

import jsonpickle
from jsonpickle import errors
from jsonpickle.pickler import Pickler
from jsonpickle.unpickler import Unpickler


class Order:
    """A business object holding a reference to shared external config."""

    def __init__(self, name, config):
        self.name = name
        self.config = config


class AddressBook:
    def __init__(self, owner):
        self.owner = owner
        self.entries = []


SHARED_CONFIG = {
    "region": "eu-west-1",
    "limits": {"qps": 100, "burst": 200},
    "features": ["beta", "audit"],
}


def make_external_ref(identifier, *tracked):
    def external_ref(obj):
        for tracked_obj in tracked:
            if obj is tracked_obj:
                return identifier
        return None

    return external_ref


class Store:
    """A resolver-backed object repository that counts resolutions."""

    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def resolver(self, identifier):
        self.calls.append(identifier)
        return self.objects[identifier]


def test_shared_external_dict_is_not_inlined_and_stays_shared():
    """Two business objects referencing one external dict."""
    config = dict(SHARED_CONFIG)
    orders = [Order("a", config), Order("b", config)]
    external_ref = make_external_ref("config://shared", config)

    encoded = jsonpickle.encode(orders, external_ref=external_ref)

    # the snapshot contains the reference, not the dictionary contents
    assert '"py/external": "config://shared"' in encoded
    assert "eu-west-1" not in encoded
    assert "qps" not in encoded
    assert "beta" not in encoded

    restored_config = dict(SHARED_CONFIG)
    store = Store({"config://shared": restored_config})
    decoded = jsonpickle.decode(encoded, external_resolver=store.resolver)

    # both orders see the same external instance
    assert decoded[0].config is decoded[1].config
    assert decoded[0].config is restored_config
    # mutating through one reference is visible through the other
    decoded[0].config["region"] = "us-east-1"
    assert decoded[1].config["region"] == "us-east-1"
    assert restored_config["region"] == "us-east-1"


def test_identifier_resolved_once_per_decode():
    config = dict(SHARED_CONFIG)
    orders = [Order("a", config), Order("b", config)]
    encoded = jsonpickle.encode(
        orders, external_ref=make_external_ref("config://shared", config)
    )
    store = Store({"config://shared": dict(SHARED_CONFIG)})

    jsonpickle.decode(encoded, external_resolver=store.resolver)

    assert store.calls == ["config://shared"]


def test_independent_decodes_do_not_share_cache():
    config = dict(SHARED_CONFIG)
    orders = [Order("a", config), Order("b", config)]
    encoded = jsonpickle.encode(
        orders, external_ref=make_external_ref("config://shared", config)
    )
    calls = []

    def fresh_resolver(identifier):
        calls.append(identifier)
        return dict(SHARED_CONFIG)

    first = jsonpickle.decode(encoded, external_resolver=fresh_resolver)
    second = jsonpickle.decode(encoded, external_resolver=fresh_resolver)

    assert calls == ["config://shared", "config://shared"]
    assert first[0].config is not second[0].config
    # each decode is still internally consistent
    assert first[0].config is first[1].config
    assert second[0].config is second[1].config


def test_explicit_contexts_carry_the_protocol():
    config = dict(SHARED_CONFIG)
    order = Order("a", config)
    pickler = Pickler(external_ref=make_external_ref("config://shared", config))
    encoded = jsonpickle.encode(order, context=pickler)
    assert "eu-west-1" not in encoded

    store = Store({"config://shared": dict(SHARED_CONFIG)})
    unpickler = Unpickler(external_resolver=store.resolver)
    decoded = jsonpickle.decode(encoded, context=unpickler)
    assert decoded.config is store.objects["config://shared"]

    # an explicit decode() argument overrides the context's resolver
    other = dict(SHARED_CONFIG)
    decoded = jsonpickle.decode(
        encoded, context=unpickler, external_resolver=lambda identifier: other
    )
    assert decoded.config is other


def test_unselected_objects_use_regular_protocol():
    config = dict(SHARED_CONFIG)
    book = AddressBook("alice")
    book.entries.append(config)
    order = Order("a", config)
    payload = {"book": book, "order": order, "plain": [1, 2, 3]}
    encoded = jsonpickle.encode(
        payload, external_ref=make_external_ref("config://shared", config)
    )
    store = Store({"config://shared": dict(SHARED_CONFIG)})

    decoded = jsonpickle.decode(encoded, external_resolver=store.resolver)

    assert isinstance(decoded["book"], AddressBook)
    assert decoded["book"].owner == "alice"
    assert decoded["plain"] == [1, 2, 3]
    assert decoded["book"].entries[0] is decoded["order"].config


def test_cycles_in_regular_graph_still_work():
    config = dict(SHARED_CONFIG)
    node = {"config": config}
    node["self"] = node
    encoded = jsonpickle.encode(
        node, external_ref=make_external_ref("config://shared", config)
    )
    store = Store({"config://shared": dict(SHARED_CONFIG)})

    decoded = jsonpickle.decode(encoded, external_resolver=store.resolver)

    assert decoded["self"] is decoded
    assert decoded["config"] is store.objects["config://shared"]


def test_external_object_is_not_recursively_expanded():
    config = dict(SHARED_CONFIG)
    config["self"] = config  # a cycle through the external object
    order = Order("a", config)
    encoded = jsonpickle.encode(
        order, external_ref=make_external_ref("config://shared", config)
    )
    assert "self" not in encoded
    store = Store({"config://shared": dict(SHARED_CONFIG)})
    decoded = jsonpickle.decode(encoded, external_resolver=store.resolver)
    assert decoded.config["region"] == "eu-west-1"


def test_missing_resolver_is_an_error():
    encoded = jsonpickle.encode(
        Order("a", SHARED_CONFIG),
        external_ref=make_external_ref("config://shared", SHARED_CONFIG),
    )
    with pytest.raises(errors.ExternalReferenceError):
        jsonpickle.decode(encoded)


def test_unknown_identifier_is_reported():
    encoded = jsonpickle.encode(
        Order("a", SHARED_CONFIG),
        external_ref=make_external_ref("config://missing", SHARED_CONFIG),
    )
    store = Store({})
    with pytest.raises(errors.ExternalReferenceNotFoundError):
        jsonpickle.decode(encoded, external_resolver=store.resolver)


def test_resolver_failure_is_reported():
    encoded = jsonpickle.encode(
        Order("a", SHARED_CONFIG),
        external_ref=make_external_ref("config://shared", SHARED_CONFIG),
    )

    def broken_resolver(identifier):
        raise RuntimeError("repository is down")

    with pytest.raises(errors.ExternalResolverError) as excinfo:
        jsonpickle.decode(encoded, external_resolver=broken_resolver)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_declared_type_mismatch_is_reported():
    encoded = jsonpickle.encode(
        Order("a", SHARED_CONFIG),
        external_ref=make_external_ref("config://shared", SHARED_CONFIG),
    )
    store = Store({"config://shared": ["not", "a", "dict"]})
    with pytest.raises(errors.ExternalReferenceTypeError):
        jsonpickle.decode(
            encoded,
            external_resolver=store.resolver,
            external_types={"config://shared": dict},
        )


def test_declared_type_accepts_matching_objects():
    encoded = jsonpickle.encode(
        Order("a", SHARED_CONFIG),
        external_ref=make_external_ref("config://shared", SHARED_CONFIG),
    )
    store = Store({"config://shared": dict(SHARED_CONFIG)})
    decoded = jsonpickle.decode(
        encoded,
        external_resolver=store.resolver,
        external_types={"config://shared": dict},
    )
    assert decoded.config is store.objects["config://shared"]

    # a bare type declaration applies to every identifier
    decoded = jsonpickle.decode(
        encoded, external_resolver=store.resolver, external_types=dict
    )
    assert decoded.config is store.objects["config://shared"]

    with pytest.raises(errors.ExternalReferenceTypeError):
        jsonpickle.decode(
            encoded, external_resolver=store.resolver, external_types=list
        )


def test_no_silent_fallback_to_class_construction():
    """A py/external payload must never degrade into a py/object restore."""
    encoded = jsonpickle.encode(
        Order("a", SHARED_CONFIG),
        external_ref=make_external_ref("config://missing", SHARED_CONFIG),
    )
    with pytest.raises(errors.ExternalReferenceNotFoundError):
        jsonpickle.decode(
            encoded,
            external_resolver=Store({}).resolver,
            classes=[Order],
            on_missing="error",
        )


def test_external_reference_inside_nested_containers():
    config = dict(SHARED_CONFIG)
    payload = {"orders": [Order("a", config)], "backup": (config,)}
    encoded = jsonpickle.encode(
        payload, external_ref=make_external_ref("config://shared", config)
    )
    store = Store({"config://shared": dict(SHARED_CONFIG)})

    decoded = jsonpickle.decode(encoded, external_resolver=store.resolver)

    assert decoded["orders"][0].config is decoded["backup"][0]
    assert store.calls == ["config://shared"]
