import time
import numpy as np

def run_steps(agent, max_steps, log_interval, log_test, eval_pth, online_data_size):
    train_logs = []
    evaluations = []
    t0 = time.time()
    agent.populate_returns(initialize=True, log_traj=True)
    agent.fill_buffer(online_data_size)
    while True:
        if log_interval and not agent.total_steps % log_interval:
            train_mean, train_median, train_min_, train_max_, test_mean, test_median, test_min_, test_max_ = agent.log_file(elapsed_time=log_interval / (time.time() - t0), test=log_interval>1 and log_test)
            train_logs.append(train_mean)
            evaluations.append(test_mean)
            
            # np.save(eval_pth + "/train_logs.npy", np.array(train_logs))
            # np.save(eval_pth + "/evaluations.npy", np.array(evaluations))
            # agent.save()
            t0 = time.time()
        
        if max_steps and agent.total_steps >= max_steps:
            break
        agent.step()
    np.save(eval_pth + "/train_logs.npy", np.array(train_logs))
    np.save(eval_pth + "/evaluations.npy", np.array(evaluations))
    agent.save()
    agent.savevis()

    agent.save_info(eval_pth + "/info_logs.pkl")
