from dataclasses import field
from functools import cached_property

import pandas as pd

import corerl.utils.list as list_u
from corerl.configs.config import config, list_
from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig, DecisionPointDetector
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import TransformConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig


@config()
class SCConfig:
    defaults: list[TransformConfig] = list_([NormalizerConfig()])
    countdown: CountdownConfig = field(default_factory=CountdownConfig)


class StateConstructor(Constructor):
    def __init__(self, tag_cfgs: list[TagConfig], cfg: SCConfig):
        self._cfg = cfg
        super().__init__(tag_cfgs)

        self._cd_adder = DecisionPointDetector(cfg.countdown)


    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]):
        return {
            tag.name: tag.state_constructor if tag.state_constructor is not None else self._cfg.defaults
            for tag in tag_cfgs
            if not tag.is_meta
        }


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if self._cd_adder is not None:
            pf = self._cd_adder(pf)

        transformed_parts, tag_names = self._transform_tags(pf, StageCode.SC)

        # put resultant data on PipeFrame
        df = pf.data.drop(tag_names, axis=1, inplace=False)
        pf.data = pd.concat([df] + transformed_parts, axis=1, copy=False)

        # guarantee an ordering over columns
        sorted_cols = list_u.multi_level_sort(
            list(pf.data.columns),
            categories=[
                # endo observations
                lambda tag: (
                    tag in self._tag_cfgs
                    and self._tag_cfgs[tag].is_endogenous
                    and not self._tag_cfgs[tag].is_meta
                ),
                # exo observations
                lambda tag: (
                    tag in self._tag_cfgs
                    and not self._tag_cfgs[tag].is_meta
                ),
                # states
                lambda tag: not (
                    tag in self._tag_cfgs
                    and self._tag_cfgs[tag].is_meta
                ),
                # meta tags should be all that are left
            ],
        )
        pf.data = pf.data.loc[:, sorted_cols]

        return pf


    @cached_property
    def columns(self):
        pf = self._probe_fake_data()
        return list(pf.data.columns)


def construct_default_sc_configs(sc_cfg: SCConfig, tag_cfgs: list[TagConfig]) -> None:
    for tag_cfg in tag_cfgs:
        if tag_cfg.state_constructor is None:
            tag_cfg.state_constructor = sc_cfg.defaults
