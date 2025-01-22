from corerl.configs.config import MISSING, config
from corerl.data_pipeline.datatypes import CallerCode, StageCode


@config()
class BaseEvalConfig:
    name: str = MISSING
    caller_codes: list[CallerCode] = MISSING
    stage_codes: list[StageCode] = MISSING
    enabled: bool = True
