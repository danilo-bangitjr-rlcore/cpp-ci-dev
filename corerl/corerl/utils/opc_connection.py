from __future__ import annotations

from uuid import UUID


def make_opc_node_id(node_id: str | int | bytes | UUID, namespace: str | int = 0):
    if isinstance(node_id, int):
        return f"ns={namespace};i={node_id}"
    elif isinstance(node_id, UUID):
        return f"ns={namespace};g={node_id}"
    elif isinstance(node_id, bytes):
        return f"ns={namespace};b={node_id}"
    else:
        return f"ns={namespace};s={node_id}"
