from typing import Any, SupportsFloat, TypeVar

T = TypeVar('T', bound=SupportsFloat | dict[str, Any])

def flatten_tree(
    tree: dict[str, T],
    prefix: str = '',
    sep: str = '_',
) -> dict[str, float]:
    result = {}
    for key, value in tree.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_tree(value, new_key, sep))
        else:
            result[new_key] = float(value)

    return result
