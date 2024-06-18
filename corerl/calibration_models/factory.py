from corerl.calibration_models.one_step import OneStep
from corerl.calibration_models.anytime import AnytimeCalibrationModel


def init_calibration_model(cfg, train_info):
    name = cfg.name
    if name == 'one_step':
        cm = OneStep(cfg, train_info)
    elif name == 'anytime':
        cm = AnytimeCalibrationModel(cfg, train_info)
    else:
        raise NotImplementedError

    return cm
