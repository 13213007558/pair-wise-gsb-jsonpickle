import builtins

import pytest

import jsonpickle
import jsonpickle.handlers
from jsonpickle import tags
from jsonpickle.inspector import InspectionError


class Root:
    pass


class Stateful:
    def __getstate__(self):
        return {"answer": 42}


class Counted:
    constructed = 0

    def __new__(cls, *args, **kwargs):
        cls.constructed += 1
        return super().__new__(cls)


class Handled:
    pass


class CountingHandler(jsonpickle.handlers.BaseHandler):
    restored = 0

    def flatten(self, obj, data):
        data["handled"] = True
        return data

    def restore(self, data):
        type(self).restored += 1
        return Handled()


def test_inspect_preserves_shared_and_cyclic_identity():
    root = Root()
    shared = []
    root.first = shared
    root.second = shared
    shared.append(root)

    graph = jsonpickle.inspect(jsonpickle.encode(root))

    first = graph.root["first"]
    assert first is graph.root["second"]
    assert first[0] is graph.root
    assert graph.get(("second",)) is first
    assert graph.get("second/0") is graph.root
    assert graph.find_path(first) == ("first",)
    assert graph.find_path(graph.root, first) == (0,)

    incoming = {(edge.source, edge.name, edge.kind) for edge in first.incoming()}
    assert incoming == {
        (graph.root, "first", "child"),
        (graph.root, "second", "reference"),
    }
    root_incoming = graph.references_to(graph.root)
    assert [(edge.source, edge.name) for edge in root_incoming] == [(first, 0)]
    assert len({node.identity for node in graph.nodes}) == len(graph.nodes)
    assert graph.root.id == graph.root.identity


def test_inspect_preserves_class_state_and_container_contents():
    graph = jsonpickle.inspect(jsonpickle.encode({"stateful": Stateful()}))

    stateful = graph.root["stateful"]
    assert stateful.kind == "object"
    assert stateful.class_name.endswith(".Stateful")
    assert stateful.state["answer"].value == 42

    containers = jsonpickle.inspect(
        {
            "list": [1, 2],
            "tuple": {tags.TUPLE: ["a", "b"]},
            "set": {tags.SET: [True]},
        }
    )
    assert [node.value for node in containers.root["list"].children()] == [1, 2]
    assert [node.value for node in containers.root["tuple"].children()] == ["a", "b"]
    assert [node.value for node in containers.root["set"].children()] == [True]


def test_inspect_does_not_import_construct_or_run_handlers(monkeypatch):
    imports = 0
    real_import = builtins.__import__

    def counting_import(name, *args, **kwargs):
        nonlocal imports
        imports += 1
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", counting_import)
    Counted.constructed = 0
    CountingHandler.restored = 0
    jsonpickle.handlers.register(Handled, CountingHandler)
    try:
        handled = jsonpickle.encode(Handled())
        counted = jsonpickle.encode(object.__new__(Counted))
        payload = {
            "handled": jsonpickle.backend.json.decode(handled),
            "counted": jsonpickle.backend.json.decode(counted),
            "missing": {tags.OBJECT: "snapshot_missing.DoesNotExist", "value": 1},
            "function": {tags.FUNCTION: "snapshot_missing.function"},
            "reduce": {
                tags.REDUCE: [
                    {tags.FUNCTION: "snapshot_missing.reduce"},
                    {tags.TUPLE: []},
                ]
            },
            "repr": {tags.REPR: "snapshot_missing/attribute"},
        }

        imports_before = imports
        graph = jsonpickle.inspect(payload)
        mode_graph = jsonpickle.decode(
            '{"missing": {"py/object": "snapshot_missing.Other"}}',
            mode="inspect",
        )

        assert imports == imports_before
        assert Counted.constructed == 0
        assert CountingHandler.restored == 0
        assert mode_graph.root["missing"].class_name == "snapshot_missing.Other"
        assert graph.root["missing"].class_name == "snapshot_missing.DoesNotExist"
        assert graph.root["function"].executable == {
            "type": "function",
            "name": "snapshot_missing.function",
        }
        assert graph.root["reduce"].kind == "reduce"
        assert graph.root["repr"].executable == {
            "type": "repr",
            "payload": "snapshot_missing/attribute",
        }
    finally:
        jsonpickle.handlers.unregister(Handled)


def test_inspect_reports_invalid_references():
    graph = jsonpickle.inspect(
        {
            "target": [],
            "missing": {tags.ID: 99},
            "negative": {tags.ID: -1},
            "wrong_type": {tags.ID: "0"},
        }
    )

    assert [item.code for item in graph.diagnostics] == ["invalid-reference"] * 3
    assert "slot 99" in str(graph.diagnostics[0])
    assert "non-negative integer" in str(graph.diagnostics[1])
    assert "found str" in str(graph.diagnostics[2])
    with pytest.raises(InspectionError, match="invalid-reference"):
        jsonpickle.inspect({"missing": {tags.ID: 99}}, strict=True)


def test_inspect_reports_duplicate_structure_identity():
    shared = []
    graph = jsonpickle.inspect({"first": shared, "second": shared})

    assert [item.code for item in graph.diagnostics] == [
        "duplicate-structure-identity"
    ]
    assert "first seen at $.first" in str(graph.diagnostics[0])
    assert graph.root["first"] is not graph.root["second"]


def test_inspect_reports_unparseable_legacy_formats():
    graph = jsonpickle.inspect(
        {
            "old_reference": {tags.REF: 0},
            "old_repr": {tags.REPR: "not-a-module-attribute"},
        }
    )

    assert [item.code for item in graph.diagnostics] == [
        "unparseable-legacy-format"
    ] * 2
    assert graph.root["old_reference"].kind == "legacy-reference"
    assert graph.root["old_repr"].kind == "repr"


def test_decode_inspect_mode_returns_graph():
    graph = jsonpickle.decode('{"value": 1}', mode="inspect")

    assert graph.root["value"].value == 1
    with pytest.raises(ValueError, match="unknown jsonpickle decode mode"):
        jsonpickle.decode("{}", mode="execute")
