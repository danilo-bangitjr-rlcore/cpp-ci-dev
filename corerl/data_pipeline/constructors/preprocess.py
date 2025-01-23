from collections.abc import Iterable
from functools import cached_property

import pandas as pd

from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms.base import InvertibleTransform
from corerl.utils.list import find


class Preprocessor(Constructor):
    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]):
        return {
            tag.name: tag.preprocess
            for tag in tag_cfgs
            # while technically we can avoid this check
            # it's epsilon more performant to skip empty preprocess steps
            # because it avoids some pandas column mutation logic
            if len(tag.preprocess) > 0 and not tag.is_meta
        }

    def __call__(self, pf: PipelineFrame):
        transformed_parts, tag_names = self._transform_tags(pf, StageCode.PREPROCESS)

        # put resultant data on PipeFrame
        df = pf.data.drop(tag_names, axis=1, inplace=False)
        pf.data = pd.concat([df] + transformed_parts, axis=1, copy=False)

        pf.data.rename(columns=lambda col: maybe_get_prefix(col, tag_names), inplace=True)
        return pf

    def inverse(self, df: pd.DataFrame):
        # since it is easier to mutate an existing df
        # make a copy so that this is still a pure function
        df = df.copy(deep=False)

        for tag in df.columns:
            xforms = self._components.get(tag)
            if xforms is None:
                continue

            for xform in reversed(xforms):
                if isinstance(xform, InvertibleTransform):
                    df[tag] = xform.invert(df[tag].to_numpy(), tag)

        return df

    @cached_property
    def columns(self):
        pf = self._probe_fake_data()
        return list(pf.data.columns)


def maybe_get_prefix(col: str, prefixes: Iterable[str]) -> str:
    prefix = find(lambda pre: col.startswith(pre), prefixes)
    return prefix if prefix is not None else col
