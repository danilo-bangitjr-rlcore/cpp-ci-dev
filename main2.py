import shelve
from datetime import UTC, datetime, timedelta

from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.datatypes import StageCode
from corerl.data_pipeline.transition_creators.anytime import AnytimeTransitionCreatorConfig


db_cfg = TagDBConfig(
    drivername="postgresql+psycopg2",
    username="postgres",
    password="password",
    ip="localhost",
    port=5432,  # default is 5432, but we want to use different port for test db
    db_name="postgres",
    sensor_table_name="public.scrubber4",
)

data_reader = DataReader(db_cfg=db_cfg)

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
names = [n.strip() for n in raw_names.splitlines() if n]
ts = data_reader.get_time_stats()



with shelve.open('raw_tag_config') as rtc:
    tag_configs = rtc['tag_configs'] if 'tag_configs' in rtc else []

    if len(tag_configs) == 0:
        for name in names:
            tag_stats = data_reader.get_tag_stats(name)
            print(name)
            print(tag_stats)

            tag_configs.append(TagConfig(
                name=name,
                bounds=(tag_stats.min, tag_stats.max),
                is_action= name.endswith("_SP")
            ))
    rtc['tag_configs'] = tag_configs

print(tag_configs)

end_date = datetime(2024, 12, 15, tzinfo=UTC)
start_date = datetime(2024, 12, 14, tzinfo=UTC)
# res = data_reader.single_aggregated_read(names, start_date, end_date)

res = data_reader.batch_aggregated_read(
    names,
    start_date,
    end_date,
    bucket_width=timedelta(hours=1)
)


print(res.T)

pipeline_config = PipelineConfig(
    tags=tag_configs,
    db= db_cfg,
    obs_interval_minutes=1.1,
    agent_transition_creator=AnytimeTransitionCreatorConfig(
        steps_per_decision=1
    )
)

pipeline = Pipeline(pipeline_config)

print(pipeline.transition_creator)

resp = pipeline(res, stages=(
    StageCode.BOUNDS,
    # StageCode.ODDITY,
    StageCode.IMPUTER,
    StageCode.RC,
    # StageCode.SC,
    StageCode.TC))
print(resp)
