from corerl.calibration_models.one_step import OneStep
from corerl.calibration_models.gru import GRUCalibrationModel
from corerl.calibration_models.simple_n_step import NStepCalibrationModel
# from corerl.calibration_models.anytime_interpolation import AnytimeCalibrationModel
from corerl.calibration_models.anytime import AnytimeCalibrationModel
# from corerl.calibration_models.anytime_interpolation_weight import AnytimeCalibrationModel

def init_calibration_model(cfg, train_info):
    name = cfg.name
    if name == 'simple':
        cm = OneStep(cfg, train_info)
    elif name == 'nstep':
        cm = NStepCalibrationModel(cfg, train_info)
    elif name == 'gru':
        cm = GRUCalibrationModel(cfg, train_info)
    elif name == 'anytime':
        cm = AnytimeCalibrationModel(cfg, train_info)
    else:
        raise NotImplementedError

    return cm
