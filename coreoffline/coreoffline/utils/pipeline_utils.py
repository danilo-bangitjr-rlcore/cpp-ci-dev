import pandas as pd
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from lib_agent.buffer.datatypes import DataMode


class StageDataCapture:
    """Captures pipeline dataframe state after each specified stage"""
    def __init__(self, pipeline: Pipeline):
        self.captured_data: dict[StageCode, list[pd.DataFrame]] = {}
        stages = pipeline.default_stages

        # Initialize empty lists for each stage
        for stage in stages:
            self.captured_data[stage] = []

        for stage in stages:
            pipeline.register_hook(
                data_modes=DataMode.OFFLINE,
                stages=stage,
                f=self.create_capture_hook(stage),
                order='post',
        )

    def create_capture_hook(self, stage: StageCode):
        """Returns a hook function that captures dataframe state after specified stage"""
        def hook(pf: PipelineFrame) -> None:
            self.captured_data[stage].append(pf.data.copy())

        return hook

    def get_concatenated_data(self, stage: StageCode) -> pd.DataFrame:
        """Returns concatenated dataframe for the specified stage"""
        if not self.captured_data[stage]:
            return pd.DataFrame()
        return pd.concat(self.captured_data[stage])
