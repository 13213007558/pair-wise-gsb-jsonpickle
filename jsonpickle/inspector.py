"""Non-executing inspection support for jsonpickle snapshots."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import tags
from .backend import json


class InspectionError(ValueError):
    """Raised when strict inspection finds malformed snapshot data."""


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    path: tuple[object, ...] = ()

    def __str__(self) -> str:
        location = format_path(self.path) if self.path else "$"
        return f"{self.code} at {location}: {self.message}"


@dataclass(frozen=True)
class Edge:
    name: object
    target: Node
    kind: str = "child"


@dataclass(frozen=True)
class IncomingEdge:
    source: Node
    name: object
    target: Node
    kind: str = "child"


@dataclass(kw_only=True, eq=False)
class Node:
    graph: ObjectGraph
    identity: int
    kind: str
    raw: Any
    path: tuple[object, ...]
    class_name: str | None = None
    container_type: str | None = None
    executable: dict[str, Any] | None = None
    value: Any = None
    slot: int | None = None
    edges: list[Edge] = field(default_factory=list)
    reference: Node | None = None
    resolved: bool = True

    @property
    def id(self) -> int:
        return self.identity

    @property
    def node_id(self) -> int:
        return self.identity

    @property
    def state(self) -> Node | None:
        return self.edge(tags.STATE)

    def edge(self, name: object) -> Node | None:
        for edge in self.edges:
            if edge.name == name:
                return edge.target
        return None

    def children(self) -> tuple[Node, ...]:
        return tuple(edge.target for edge in self.edges)

    def outgoing(self) -> tuple[Edge, ...]:
        return tuple(self.edges)

    def incoming(self) -> tuple[IncomingEdge, ...]:
        return self.graph.incoming_for(self)

    def referrers(self) -> tuple[Node, ...]:
        return tuple(edge.source for edge in self.graph.incoming_for(self))

    def get(self, path: object | Sequence[object] = ()) -> Node:
        node = self
        for name in normalize_path(path):
            node = node[name]
        return node

    def __getitem__(self, name: object) -> Node:
        return self.graph.follow(self, name)

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.identity,
            "kind": self.kind,
            "path": list(self.path),
        }
        if self.class_name is not None:
            result["class"] = self.class_name
        if self.container_type is not None:
            result["container"] = self.container_type
        if self.executable is not None:
            result["executable"] = self.executable
        if self.value is not None:
            result["value"] = self.value
        if self.state is not None:
            result["state"] = self.state.identity
        if self.edges:
            result["edges"] = [
                {"name": edge.name, "id": edge.target.identity, "kind": edge.kind}
                for edge in self.edges
            ]
        if self.reference is not None:
            result["reference"] = self.reference.identity
        if not self.resolved:
            result["resolved"] = False
        return result


@dataclass(kw_only=True)
class ObjectGraph:
    root: Node
    diagnostics: tuple[Diagnostic, ...] = ()
    incoming_edges: list[IncomingEdge] = field(default_factory=list)
    _nodes: dict[int, Node] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nodes = sorted(self._iter_nodes(), key=lambda node: node.identity)
        object.__setattr__(self, "_nodes", {node.identity: node for node in nodes})
        edges = [
            IncomingEdge(node, edge.name, edge.target, edge.kind)
            for node in nodes
            for edge in node.edges
        ]
        object.__setattr__(self, "incoming_edges", edges)

    def _iter_nodes(self) -> Iterator[Node]:
        seen: set[int] = set()
        pending = [self.root]
        while pending:
            node = pending.pop()
            if node.identity in seen:
                continue
            seen.add(node.identity)
            yield node
            pending.extend(edge.target for edge in node.edges)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(sorted(self._nodes.values(), key=lambda node: node.identity))

    def node(self, identity: int) -> Node:
        return self._nodes[identity]

    def get_node(self, identity: int) -> Node | None:
        return self._nodes.get(identity)

    def follow(self, node: Node, name: object) -> Node:
        for edge in node.edges:
            if edge.name == name:
                return edge.target
        raise KeyError(name)

    def get(self, path: object | Sequence[object] = ()) -> Node:
        node = self.root
        for name in normalize_path(path):
            node = node[name]
        return node

    def references_to(self, node: Node) -> tuple[IncomingEdge, ...]:
        return self.incoming_for(node)

    def incoming_for(self, node: Node) -> tuple[IncomingEdge, ...]:
        return tuple(edge for edge in self.incoming_edges if edge.target is node)

    def find_path(
        self, target: Node, source: Node | None = None
    ) -> tuple[object, ...] | None:
        start = source or self.root
        if start is target:
            return ()
        pending: deque[tuple[Node, tuple[object, ...]]] = deque([(start, ())])
        seen = {start.identity}
        while pending:
            current, path = pending.popleft()
            for edge in current.edges:
                next_path = (*path, edge.name)
                if edge.target is target:
                    return next_path
                if edge.target.identity not in seen:
                    seen.add(edge.target.identity)
                    pending.append((edge.target, next_path))
        return None

    def path_to(
        self, target: Node, source: Node | None = None
    ) -> tuple[object, ...] | None:
        return self.find_path(target, source)

    def describe(self) -> dict[str, Any]:
        return {
            "root": self.root.identity,
            "nodes": [node.describe() for node in self.nodes],
            "diagnostics": [str(item) for item in self.diagnostics],
        }


def format_path(path: Sequence[object]) -> str:
    result = "$"
    for item in path:
        if isinstance(item, int) and not isinstance(item, bool):
            result += f"[{item}]"
        else:
            result += f".{item}"
    return result


def normalize_path(path: object | Sequence[object]) -> tuple[object, ...]:
    if isinstance(path, str):
        if path in ("", "$", "/"):
            return ()
        if "/" not in path:
            return (path,)
        parts = [part for part in path.removeprefix("$/").removeprefix("/").split("/") if part]
        return tuple(
            int(part) if part.isdecimal() else part
            for part in parts
        )
    if isinstance(path, bytes) or not isinstance(path, Sequence):
        return (path,)
    return tuple(path)


def primitive_kind(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) in (bool, int, float, str):
        return type(value).__name__
    return "unsupported"


class _GraphBuilder:
    def __init__(self, *, keys: bool = True, strict: bool = False) -> None:
        self.keys = keys
        self.strict = strict
        self.next_identity = 0
        self.next_slot = 0
        self.nodes: list[Node] = []
        self.active_containers: dict[int, Node] = {}
        self.completed_containers: dict[int, Node] = {}
        self.slots: list[Node] = []
        self.pending_references: list[tuple[Node, Any, tuple[object, ...]]] = []
        self.diagnostics: list[Diagnostic] = []

    def diagnostic(self, code: str, message: str, path: Sequence[object]) -> None:
        item = Diagnostic(code, message, tuple(path))
        self.diagnostics.append(item)
        if self.strict:
            raise InspectionError(str(item))

    def register_slot(self, node: Node) -> None:
        node.slot = self.next_slot
        self.slots.append(node)
        self.next_slot += 1

    def build_node(self, value: Any, path: tuple[object, ...]) -> Node:
        if isinstance(value, (dict, list)):
            existing = self.active_containers.get(id(value))
            if existing is not None:
                self.diagnostic(
                    "duplicate-structure-identity",
                    "the decoded structure refers to itself without a py/id "
                    f"reference (first seen at {format_path(existing.path)})",
                    path,
                )
                return existing
            if id(value) in self.completed_containers:
                existing = self.completed_containers[id(value)]
                self.diagnostic(
                    "duplicate-structure-identity",
                    "the same decoded container occurs at more than one path "
                    "without a py/id reference; a separate descriptor is used "
                    f"instead of guessing shared identity (first seen at "
                    f"{format_path(existing.path)})",
                    path,
                )

        node = Node(
            graph=None,  # type: ignore[arg-type]
            identity=self.next_identity,
            kind="value",
            raw=value,
            path=path,
        )
        self.next_identity += 1
        self.nodes.append(node)
        if isinstance(value, (dict, list)):
            self.active_containers[id(value)] = node

        try:
            if type(value) is list:
                self.register_slot(node)
                node.kind = "list"
                node.container_type = "list"
                for index, item in enumerate(value):
                    self.add_child(node, index, item, path)
            elif type(value) is dict:
                self.populate_dict(node, value, path)
            else:
                node.kind = primitive_kind(value)
                node.value = value
        finally:
            if isinstance(value, (dict, list)):
                self.completed_containers.setdefault(id(value), node)
                self.active_containers.pop(id(value), None)
        return node

    def add_child(
        self,
        parent: Node,
        name: object,
        value: Any,
        path: tuple[object, ...],
        *,
        kind: str = "child",
    ) -> Node:
        child = self.build_node(value, path + (name,))
        parent.edges.append(Edge(name, child, kind))
        return child

    def populate_dict(
        self, node: Node, value: dict[Any, Any], path: tuple[object, ...]
    ) -> None:
        if tags.ID in value:
            node.kind = "reference"
            node.resolved = False
            self.pending_references.append((node, value[tags.ID], path))
            return

        if tags.TUPLE in value:
            self.tagged_container(node, value, tags.TUPLE, "tuple", path)
            return
        if tags.SET in value:
            self.tagged_container(node, value, tags.SET, "set", path)
            return
        if tags.ITERATOR in value:
            self.register_slot(node)
            self.tagged_container(node, value, tags.ITERATOR, "iterator", path)
            return
        if tags.BYTEARRAY in value:
            self.populate_bytearray(node, value, path)
            return
        if tags.B64 in value:
            node.kind = "bytes"
            node.executable = {
                "type": "bytes",
                "encoding": "base64",
                "payload": value[tags.B64],
            }
            return
        if tags.B85 in value:
            node.kind = "bytes"
            node.executable = {
                "type": "bytes",
                "encoding": "base85",
                "payload": value[tags.B85],
            }
            return
        if tags.REF in value:
            self.register_slot(node)
            node.kind = "legacy-reference"
            node.executable = {
                "type": "legacy-reference",
                "tag": tags.REF,
                "payload": value[tags.REF],
            }
            self.diagnostic(
                "unparseable-legacy-format",
                "py/ref is an old reference format and is not interpreted by inspect mode",
                path,
            )
            return
        if tags.OBJECT in value:
            self.register_slot(node)
            self.populate_object(node, value, path)
            return
        if tags.TYPE in value:
            name = value[tags.TYPE]
            node.kind = "type"
            node.class_name = name if isinstance(name, str) else str(name)
            node.executable = {"type": "type", "name": name}
            return
        if tags.REDUCE in value:
            self.register_slot(node)
            self.populate_reduce(node, value, path)
            return
        if tags.FUNCTION in value:
            name = value[tags.FUNCTION]
            node.kind = "function"
            node.class_name = name if isinstance(name, str) else str(name)
            node.executable = {"type": "function", "name": name}
            return
        if tags.MODULE in value:
            self.register_slot(node)
            name = value[tags.MODULE]
            node.kind = "module"
            node.class_name = name if isinstance(name, str) else str(name)
            node.executable = {"type": "module", "name": name}
            return
        if tags.REPR in value:
            self.register_slot(node)
            self.populate_repr(node, value, path)
            return

        self.register_slot(node)
        node.kind = "dict"
        node.container_type = "dict"
        items = list(value.items())
        for key, item in items:
            if self.keys and isinstance(key, str) and key.startswith(tags.JSON_KEY):
                continue
            self.add_child(node, key, item, path)
        if self.keys:
            for key, item in items:
                if not (isinstance(key, str) and key.startswith(tags.JSON_KEY)):
                    continue
                self.inspect_json_key(key, path)
                self.add_child(node, key, item, path)
        else:
            for key, item in items:
                if isinstance(key, str) and key.startswith(tags.JSON_KEY):
                    self.add_child(node, key, item, path)

    def tagged_container(
        self,
        node: Node,
        value: dict[Any, Any],
        tag: str,
        kind: str,
        path: tuple[object, ...],
    ) -> None:
        node.kind = kind
        node.container_type = kind
        payload = value[tag]
        if not isinstance(payload, list):
            self.diagnostic(
                "invalid-tag-payload",
                f"{tag} must contain a list, found {type(payload).__name__}",
                path,
            )
            node.executable = {"type": kind, "tag": tag, "payload": payload}
            return
        for index, item in enumerate(payload):
            self.add_child(node, index, item, path)

    def populate_bytearray(
        self, node: Node, value: dict[Any, Any], path: tuple[object, ...]
    ) -> None:
        payload = value[tags.BYTEARRAY]
        node.kind = "bytearray"
        if isinstance(payload, dict) and tags.B64 in payload:
            node.executable = {
                "type": "bytearray",
                "encoding": "base64",
                "payload": payload[tags.B64],
            }
        elif isinstance(payload, dict) and tags.B85 in payload:
            node.executable = {
                "type": "bytearray",
                "encoding": "base85",
                "payload": payload[tags.B85],
            }
        else:
            node.executable = {"type": "bytearray", "payload": payload}
            self.diagnostic(
                "invalid-tag-payload",
                "py/bytearray must contain a py/b64 or py/b85 payload",
                path,
            )

    def inspect_json_key(self, key: str, path: tuple[object, ...]) -> None:
        try:
            payload = json.decode(key[len(tags.JSON_KEY) :])
        except Exception as error:
            self.diagnostic(
                "unparseable-json-key",
                f"json:// key could not be parsed without execution: {error}",
                path + (key,),
            )
            return
        self.build_node(payload, path + (key,))

    def populate_object(
        self, node: Node, value: dict[Any, Any], path: tuple[object, ...]
    ) -> None:
        class_name = value.get(tags.OBJECT)
        if not isinstance(class_name, str):
            self.diagnostic(
                "invalid-class-name",
                "py/object must name a class with a string",
                path + (tags.OBJECT,),
            )
            class_name = str(class_name)
        node.kind = "object"
        node.class_name = class_name

        consumed: set[object] = set()
        factory_key: object = None
        if tags.DEFAULT_FACTORY in value:
            factory_key = tags.DEFAULT_FACTORY
        elif "default_factory" in value:
            factory_key = "default_factory"
        if factory_key is not None:
            consumed.add(factory_key)
            node.executable = {"type": "factory", "name": factory_key}
            self.add_child(node, factory_key, value[factory_key], path)

        argument_key: object = None
        if tags.NEWARGSEX in value:
            argument_key = tags.NEWARGSEX
        elif tags.NEWARGS in value:
            argument_key = tags.NEWARGS
        elif tags.INITARGS in value:
            argument_key = tags.INITARGS
        if argument_key is not None:
            consumed.add(argument_key)
            descriptor = node.executable or {"type": "constructor"}
            descriptor["arguments"] = argument_key
            node.executable = descriptor
            self.add_child(node, argument_key, value[argument_key], path)

        for key, item in value.items():
            if key in tags.RESERVED or key in consumed:
                continue
            if self.keys and isinstance(key, str) and key.startswith(tags.JSON_KEY):
                self.inspect_json_key(key, path)
            self.add_child(node, key, item, path)
        if tags.SEQ in value:
            self.add_child(node, tags.SEQ, value[tags.SEQ], path)
        if tags.STATE in value:
            self.add_child(node, tags.STATE, value[tags.STATE], path)

    def populate_reduce(
        self, node: Node, value: dict[Any, Any], path: tuple[object, ...]
    ) -> None:
        payload = value[tags.REDUCE]
        node.kind = "reduce"
        node.class_name = "reduce"
        if not isinstance(payload, list):
            self.diagnostic(
                "invalid-tag-payload",
                f"{tags.REDUCE} must contain a list, found {type(payload).__name__}",
                path,
            )
            node.executable = {"type": "reduce", "payload": payload}
            return
        node.executable = {"type": "reduce"}
        for index, item in enumerate(payload):
            self.add_child(node, index, item, path)

    def populate_repr(
        self, node: Node, value: dict[Any, Any], path: tuple[object, ...]
    ) -> None:
        payload = value[tags.REPR]
        node.kind = "repr"
        node.class_name = payload if isinstance(payload, str) else str(payload)
        node.executable = {"type": "repr", "payload": payload}
        if not isinstance(payload, str) or payload.count("/") != 1:
            self.diagnostic(
                "unparseable-legacy-format",
                "py/repr must use the module/attribute form; legacy executable reprs "
                "are shown as data and are not evaluated",
                path,
            )
            return
        module, attribute = payload.split("/")
        if not module or not attribute:
            self.diagnostic(
                "unparseable-legacy-format",
                "py/repr must name both a module and an attribute",
                path,
            )

    def resolve_references(self, root: Node) -> Node:
        for reference, reference_id, path in self.pending_references:
            if isinstance(reference_id, bool) or not isinstance(reference_id, int):
                self.diagnostic(
                    "invalid-reference",
                    f"{tags.ID} must be a non-negative integer, found "
                    f"{type(reference_id).__name__}",
                    path,
                )
                continue
            if reference_id < 0:
                self.diagnostic(
                    "invalid-reference",
                    f"{tags.ID} must be a non-negative integer, found "
                    f"{reference_id}",
                    path,
                )
                continue
            if reference_id >= len(self.slots):
                self.diagnostic(
                    "invalid-reference",
                    f"{tags.ID} refers to slot {reference_id}, but this snapshot "
                    f"only defines {len(self.slots)} referenceable slot(s)",
                    path,
                )
                continue

            target = self.slots[reference_id]
            reference.reference = target
            reference.resolved = True
            for node in self.nodes:
                node.edges = [
                    self._rewrite_reference_edge(edge, reference, target)
                    for edge in node.edges
                ]

        for node in self.nodes:
            node.edges = [
                edge
                if not (
                    edge.target.kind == "reference" and not edge.target.resolved
                )
                else Edge(edge.name, edge.target, "invalid-reference")
                for edge in node.edges
            ]

        return root.reference if root.kind == "reference" and root.reference else root

    def _rewrite_reference_edge(
        self, edge: Edge, reference: Node, target: Node
    ) -> Edge:
        if edge.target is reference:
            return Edge(edge.name, target, "reference")
        return edge

    def build(self, root_value: Any) -> ObjectGraph:
        root = self.build_node(root_value, ())
        root = self.resolve_references(root)
        graph = ObjectGraph(
            root=root,
            diagnostics=tuple(self.diagnostics),
        )
        for node in self.nodes:
            node.graph = graph
        return graph


def inspect(
    snapshot: str | bytes | bytearray | Any,
    *,
    keys: bool = True,
    strict: bool = False,
) -> ObjectGraph:
    """Build a descriptive object graph without importing or executing data.

    Malformed but syntactically valid snapshots are represented as graph nodes
    and reported in ``ObjectGraph.diagnostics``. Pass ``strict=True`` to raise
    :class:`InspectionError` for the first such diagnostic instead.
    """
    if isinstance(snapshot, ObjectGraph):
        return snapshot
    if isinstance(snapshot, (bytes, bytearray)):
        snapshot = bytes(snapshot).decode("utf-8")
    if isinstance(snapshot, str):
        try:
            data = json.decode(snapshot)
        except Exception as error:
            raise InspectionError(f"unparseable-json at $: {error}") from error
    else:
        data = snapshot
    return _GraphBuilder(keys=keys, strict=strict).build(data)


inspect_snapshot = inspect
