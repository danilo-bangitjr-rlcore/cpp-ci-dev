from corerl.data_pipeline.outlier_detectors.base import BaseOutlierDetector, BaseOutlierDetectorConfig, outlier_group
import corerl.data_pipeline.outlier_detectors.exp_moving_detector  # noqa: F401
import corerl.data_pipeline.outlier_detectors.identity  # noqa: F401


def init_outlier_detector(cfg: BaseOutlierDetectorConfig) -> BaseOutlierDetector:
    return outlier_group.dispatch(cfg)
