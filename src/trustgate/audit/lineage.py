"""Data lineage tracking across ingest, profiling, damage, repair, and certification."""

import hashlib
from dataclasses import dataclass, field
from typing import Any

import polars as pl


def hash_dataframe(df: pl.DataFrame) -> str:
    """Compute deterministic SHA-256 hash of DataFrame contents."""
    # Write to memory as IPC/Arrow stream and hash bytes
    try:
        data_bytes = df.write_ipc(None)
        return hashlib.sha256(data_bytes).hexdigest()
    except Exception:
        # Fallback to string repr hash
        return hashlib.sha256(str(df.head(100)).encode("utf-8")).hexdigest()


@dataclass
class LineageNode:
    node_id: str
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    source_id: str
    target_id: str
    relation: str


class LineageGraph:
    """Graph structure recording the complete transformation provenance of a data batch."""

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[LineageEdge] = []

    def add_node(self, node_id: str, stage: str, metadata: dict[str, Any]) -> None:
        self.nodes[node_id] = LineageNode(node_id=node_id, stage=stage, metadata=metadata)

    def add_edge(self, source_id: str, target_id: str, relation: str = "transformed_to") -> None:
        self.edges.append(LineageEdge(source_id=source_id, target_id=target_id, relation=relation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "nodes": [
                {"id": n.node_id, "stage": n.stage, "metadata": n.metadata}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source_id, "target": e.target_id, "relation": e.relation}
                for e in self.edges
            ],
        }

    def to_graphviz(self) -> str:
        """Render lineage graph in Graphviz DOT format."""
        lines = [
            f'digraph "Lineage_{self.batch_id}" {{',
            "  rankdir=LR;",
            "  node [shape=box, style=rounded];",
        ]
        for node in self.nodes.values():
            lbl = f"{node.stage}\\n({node.node_id})"
            lines.append(f'  "{node.node_id}" [label="{lbl}"];')
        for edge in self.edges:
            lines.append(f'  "{edge.source_id}" -> "{edge.target_id}" [label="{edge.relation}"];')
        lines.append("}")
        return "\n".join(lines)
