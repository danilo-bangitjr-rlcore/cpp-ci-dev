from itertools import islice

from asyncua.sync import Client

from corerl.utils.opc_connection import sync_browse_opc_nodes


def read_opc(url: str = "opc.tcp://localhost:4840"):
    with Client(url) as client:
        root = client.nodes.root
        opc_structure = sync_browse_opc_nodes(client, root)
        return opc_structure


def get_variables_from_dict(opc_structure: dict):
    _variables = []

    def traverse(node: dict, path: str = "", parent_key: str = ""):
        if "val" in node.keys():
            node["path"] = path
            node["key"] = parent_key
            _variables.append(node)
        else:

            if path == "":
                path = parent_key
            else:
                path = path + f"/{parent_key}"

            for key, value in node.items():
                # Variables named Opc.Ua are too long.
                # Hardcoding skipping those variables.
                if key == "Opc.Ua":
                    continue
                traverse(value, path=path, parent_key=key)

    traverse(opc_structure)
    return _variables


if __name__ == "__main__":
    opc_structure = read_opc()
    variables = get_variables_from_dict(opc_structure)
    for line in islice(variables, 10):
        print(line)
