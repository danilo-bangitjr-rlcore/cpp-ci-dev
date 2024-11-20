from corerl.data_pipeline.pipeline import PipelineConfig, Pipeline
from corerl.data_pipeline.datatypes import PipelineFrame

def test_construct_pipeline():
    pf_cfg = PipelineConfig()
    pipeline = Pipeline(pf_cfg)

