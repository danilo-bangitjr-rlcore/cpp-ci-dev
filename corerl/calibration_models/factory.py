from corerl.calibration_models.simple_one_step import SimpleCalibrationModel
from corerl.calibration_models.simple_gru_warmup import GRUCalibrationModel
from corerl.calibration_models.simple_n_step import NStepCalibrationModel

def init_calibration_model(cfg, train_info):
    name = cfg.name
    if name == 'simple':
        cm = SimpleCalibrationModel(cfg, train_info)
    elif name == 'nstep':
        cm = NStepCalibrationModel(cfg, train_info)
    elif name == 'gru':
        cm = GRUCalibrationModel(cfg, train_info)
    else:
        raise NotImplementedError

    return cm
