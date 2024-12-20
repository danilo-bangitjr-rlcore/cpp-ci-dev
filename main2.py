"""
Sanity check script. Trying to call the pipeline directly with
EPCOR Scrubber data directly from TSDB.
"""

import shelve
from datetime import UTC, datetime, timedelta

from omegaconf import DictConfig

from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.datatypes import StageCode
from corerl.data_pipeline.transition_creators.anytime import AnytimeTransitionCreatorConfig
from corerl.data_pipeline.state_constructors.sc import SCConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.environment.reward.scrubber import ScrubberReward
from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig

raw_names = """
AI0879A
AI0879B
AI0879C
AI0897B
AIC3730_MODE
AIC3730_OUT
AIC3730_PV
AIC3730_SP
AIC3730_SPRL
AIC3731_MODE
AIC3731_OUT
AIC3731_PV
AIC3731_SP
AIC3731_SPRL
FI0872
FIC3734_MODE
FIC3734_OUT
FIC3734_PV
FIC3734_SP
FIC3734_SPRL
FI_0871
FV3735_PV
LI3734
M3730A_PV
M3730B_PV
M3731A_PV
M3731B_PV
M3739_PV
PDIC3738_MODE
PDIC3738_OUT
PDIC3738_PV
PDIC3738_SP
PDIC3738_SPRL
PI0169
RLCORE_OPCUA_WD
TI0880
WATCHDOG
"""

def main():
    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=5432,  # default is 5432, but we want to use different port for test db
        db_name="postgres",
        sensor_table_name="scrubber4",
        sensor_table_schema="public"
    )

    data_reader = DataReader(db_cfg=db_cfg)
    names = [n.strip() for n in raw_names.splitlines() if n.strip()]

    with shelve.open('raw_tag_config') as rtc:
        tag_configs = rtc['tag_configs'] if 'tag_configs' in rtc else []

        if len(tag_configs) == 0:
            for name in names:
                tag_stats = data_reader.get_tag_stats(name)
                # print(name)
                # print(tag_stats)

                tag_configs.append(TagConfig(
                    name=name,
                    bounds=(tag_stats.min, tag_stats.max),
                    outlier=IdentityFilterConfig(
                        name="identity",
                    ),
                    is_action= name.endswith("_SP"),
                ))
                rtc['tag_configs'] = tag_configs

    end_date = datetime(2024, 12, 15, tzinfo=UTC)
    start_date = datetime(2024, 12, 14, tzinfo=UTC)

    # res = data_reader.single_aggregated_read(names, start_date, end_date)
    res = data_reader.batch_aggregated_read(
        names,
        start_date,
        end_date,
        bucket_width=timedelta(hours=1),
    )

    # this is needed to set transitions, if reward is NaN transitions do not get emitted.
    reward_cfg = DictConfig({
        'reward_tags': None,
        'steps_per_interaction': 1,
        'only_dp_transitions': False,
        'type': 'mean_efficiency_cost',
    })
    reward = ScrubberReward(reward_cfg)

    # appears to only require
    # AI0879C, AIC3731_OUT, AIC3730_OUT
    # https://github.com/rlcoretech/core-rl/blob/85c3ee1e3f9ef35fd5522fb87faf7e45a0a6aa68/corerl/environment/reward/scrubber.py#L62
    res['reward'] = reward(None, df_rows=res)

    pipeline_config = PipelineConfig(
        tags=tag_configs,
        db=db_cfg,
        obs_interval_minutes=1.1,
        # agent_transition_creator=AnytimeTransitionCreatorConfig(
        #     steps_per_decision=1,
        #     gamma=1,
        # ),
        state_constructor=SCConfig(
            countdown=CountdownConfig(
                 # without specifying an int, this will fail because hydra cannot
                 # interpolate in this context
                action_period=1,
                kind='no_countdown'
            ),
            defaults=[]
        )
    )

    pipeline = Pipeline(pipeline_config)

    resp = pipeline(
        res,
        stages=(
            StageCode.BOUNDS,
            StageCode.ODDITY,
            StageCode.IMPUTER,
            StageCode.RC,
            StageCode.SC, # appears to reorder the dataframe, putting actions first?
            StageCode.TC
        )
    )
    print(
        resp.df,
        len(resp.transitions) if resp.transitions else None
    )
    print(pipeline.get_state_action_dims())
    print(res.shape)
    print(len(names))


if __name__ == "__main__":
    main()

