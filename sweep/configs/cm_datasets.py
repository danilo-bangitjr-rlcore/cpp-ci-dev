from copy import deepcopy

FILENAMES = ["exp.csv", "f_1.csv", "f_b_1.csv", "f_2.csv", "m_b_1.csv", "m_1.csv",
             "m_b_2.csv", "m_2.csv", "m_3.csv", "rosie.csv", "rosie_2_alert.csv"]


def get_train_data(test_name):
    test_name = test_name[0]
    test_name = test_name.strip("\\['").strip("'\\]")
    train_files = deepcopy(FILENAMES)
    train_files.remove(test_name)
    return [train_files]


def get_test_data():
    test_files = deepcopy(FILENAMES)
    l = [[t] for t in test_files]

    return l


SWEEP_PARAMS = {
    # 'experiment.seed': list(range(1)),
    # 'calibration_model': ['knn', 'anytime', 'one_step'],
    # 'calibration_model.learn_metric': lambda d: [False, True] if d['calibration_model'] == 'knn' else None,
    # 'calibration_model.num_neighbors': lambda d: [3, 5] if d['calibration_model'] == 'knn' else None,
    # 'calibration_model.output_dim': lambda d: [3, 5, 10] if d['calibration_model'] == 'knn' else None,
    'data_loader.test_filenames': get_test_data(),
    'data_loader.train_filenames': lambda d: get_train_data(d['data_loader.test_filenames']),
}

