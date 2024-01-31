import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


def run_steps(agent, max_steps, log_interval, log_test, eval_pth, online_data_size, agent_step):
    start_time = time.time()
    train_logs = []
    evaluations = []
    t0 = time.time()
    # agent.populate_returns(initialize=True, log_traj=True)
    # agent.fill_buffer(online_data_size)
    while True:
        if agent.get_ep_returns_queue_train().shape[0] != 0: # only log when returns have been recorded
            train_mean, train_median, train_min_, train_max_, test_mean, test_median, test_min_, test_max_ = agent.log_file(elapsed_time=log_interval / (time.time() - t0), test=log_interval>1 and log_test)
            train_logs.append(train_mean)
            evaluations.append(test_mean)
        
        t0 = time.time()
        if max_steps and agent.total_steps >= max_steps:
            break
        
        agent_step()
    
    np.save(eval_pth + "/train_logs.npy", np.array(train_logs))
    np.save(eval_pth + "/evaluations.npy", np.array(evaluations))
    agent.save_info(eval_pth + "/info_logs.pkl")
    agent.save()
    agent.savevis()

    end_time = time.time()
    print("Total Time:", str(end_time - start_time))
    
    
def vis_reward(env, title, clip=None):
    if os.path.isfile(title+".pkl"):
        with open(title + ".pkl", 'rb') as f:
            true_reward = pkl.load(f)
        r_cover_space = true_reward["reward"]
        coord = true_reward["coord"]
    else:
        action_cover_space, heatmap_shape = env.get_action_samples(n=100)
        r_cover_space = []
        for i in range(len(action_cover_space)):
            o, _ = env.reset()
            _, r, done, trunc, _ = env.step(action_cover_space[i])
            r_cover_space.append(r)
        r_cover_space = np.array(r_cover_space).reshape(heatmap_shape)
        coord = np.array([action_cover_space[:, d].reshape(heatmap_shape) for d in range(action_cover_space.shape[1])])
        true_reward = {
            "reward": r_cover_space,
            "coord": coord
        }
        with open(title+".pkl", 'wb') as f:
            pkl.dump(true_reward, f)

    if clip is not None:
        r_cover_space = np.clip(r_cover_space, clip[0], clip[1])
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    im = axs.imshow(r_cover_space)
    idx = np.arange(0, coord.shape[2], 10)
    xlb = coord[0, 0, :][idx]
    ylb = coord[1, :, 0][idx]
    axs.set_xticks(idx, labels=["{:.2f}".format(x) for x in xlb], rotation=90)
    axs.set_xlabel("dim 0")
    # axs.set_xlabel("Kp")
    axs.set_yticks(idx, labels=["{:.2f}".format(x) for x in ylb])
    axs.set_ylabel("dim 1")
    # axs.set_ylabel("tau")
    plt.colorbar(im, ax=axs)
    plt.tight_layout()
    plt.savefig(title+".png", dpi=300, bbox_inches='tight')
