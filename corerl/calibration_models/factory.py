from corerl.calibration_models.simple import SimpleCalibrationModel
from corerl.calibration_models.simple_n_step import SimpleNStepCalibrationModel
from corerl.calibration_models.simple_gru import SimpleGRUCalibrationModel

def init_calibration_model(cfg, kwargs):
    name = cfg.name
    if name == 'simple':
        cm = SimpleGRUCalibrationModel(cfg, **kwargs)
    else:
        raise NotImplementedError

    return cm
