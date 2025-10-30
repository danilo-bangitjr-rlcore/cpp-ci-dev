from collections.abc import Callable
from typing import Any

from pydantic_extra_types.semantic_version import SemanticVersion


class Mapper[V: object]:
    def __init__(self, latest_schema: type[V]) -> None:
        self._transforms: dict[
            tuple[SemanticVersion, SemanticVersion],
            Callable[[object], object],
        ] = {}

        self._latest_schema = latest_schema

    def register_transform(
        self,
        version_from: SemanticVersion,
        version_to: SemanticVersion,
    ):
        def decorator[F: Callable[[Any], object]](
            transform_func: F,
        ) -> F:
            self._transforms[(version_from, version_to)] = transform_func
            return transform_func

        return decorator

    def get_latest(self, cfg: object) -> V:
        current_version = getattr(cfg, 'schema_version', SemanticVersion(0, 0, 0))

        for (version_from, version_to), transform in self._transforms.items():
            if current_version.major == version_from.major:
                cfg = transform(cfg)
                current_version = version_to

        assert isinstance(cfg, self._latest_schema)
        return cfg
