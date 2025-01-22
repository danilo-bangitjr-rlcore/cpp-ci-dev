from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import NormalizerConfig
from corerl.utils.list import find_instance
from corerl.utils.maybe import Maybe


def default_configs_from_zones(tag_cfgs: list[TagConfig]):
    for tag_cfg in tag_cfgs:
        _default_normalization_config(tag_cfg)


def _default_normalization_config(cfg: TagConfig):
    # each bound type is fully optional
    # prefer to use red zone, fallback to black zone then yellow
    # finally fallback to learning online if no other options
    # available
    lo = (
        Maybe[float](cfg.red_bounds and cfg.red_bounds[0])
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[0])
        .otherwise(lambda: cfg.yellow_bounds and cfg.yellow_bounds[0])
    )

    hi = (
        Maybe[float](cfg.red_bounds and cfg.red_bounds[1])
        .otherwise(lambda: cfg.operating_range and cfg.operating_range[1])
        .otherwise(lambda: cfg.yellow_bounds and cfg.yellow_bounds[1])
    )

    # although each constructor type may _also_ have a normalizer
    # only automatically set the preprocessor normalizer bounds
    norm_cfg = find_instance(NormalizerConfig, cfg.preprocess)
    if norm_cfg:
        norm_cfg.min = (
            Maybe(norm_cfg.min)
            .flat_otherwise(lambda: lo)
            .unwrap()
        )

        norm_cfg.max = (
            Maybe(norm_cfg.max)
            .flat_otherwise(lambda: hi)
            .unwrap()
        )

        assert norm_cfg.from_data or (norm_cfg.min is not None and norm_cfg.max is not None), \
            'Normalizer config must have min and max if from_data=False'
