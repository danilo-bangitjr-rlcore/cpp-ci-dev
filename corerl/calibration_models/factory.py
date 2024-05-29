from corerl.calibration_models.simple import SimpleCalibrationModel
from corerl.calibration_models.simple_gru_warmup import GRUCalibrationModel


def init_calibration_model(cfg, kwargs):
    name = cfg.name
    if name == 'simple':
        cm = SimpleCalibrationModel(cfg, **kwargs)
    if name == 'gru':
        cm = GRUCalibrationModel(cfg, **kwargs)
    else:
        raise NotImplementedError

    return cm
