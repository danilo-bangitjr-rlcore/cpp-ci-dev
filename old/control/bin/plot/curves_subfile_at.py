import os
from curves import DATAROOT, sweep_offline, reproduce_demo, best_offline
from curves import visualize_training_info


def sweep_parameter():
    # pth_base = DATAROOT + "output/test_v0/ReactorEnv/With_LR_Param_Baseline/Param_Sweep/{}/"
    # pth_base = DATAROOT + "output/test_v0/Cont-CC-PermExDc-v0/With_LR_Param_Baseline/Episodic/Param_Sweep/{}/"
    # pth_base = DATAROOT + "output/test_v0/NonContexTT/Noncontext_PID_Alpha_Beta_Above_One_Buffer_Prefill/{}/"
    pth_base = DATAROOT + "output/test_v0/ReactorEnv/Param_Sweep/{}/"
    #sweep_offline(pth_base, "GAC")
    #sweep_offline(pth_base, "SAC")
    #sweep_offline(pth_base, "Reinforce")
    #sweep_offline(pth_base, "ETC")
    sweep_offline(pth_base, "GAC")

def compare_algorithms():
    """
    Reactor Environment. Compare GAC, SAC, and REINFORCE
    SHAREPATH = "output/test_v0/ReactorEnv/With_LR_Param_Baseline/Param_Sweep/"
    pths = {
        "SAC": [DATAROOT + SHAREPATH + "SAC/param_20/", "C0", 5],
        "Reinforce": [DATAROOT + SHAREPATH + "Reinforce/param_14/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_91", "C1", 1],
    }
    best_offline(pths, "ReactorEnv_Baseline", ylim=[-1500, 0])

    SHAREPATH = "output/test_v0/Cont-CC-PermExDc-v0/With_LR_Param_Baseline/Episodic/Param_Sweep/"
    pths = {
        "SAC": [DATAROOT + SHAREPATH + "SAC/param_37/", "C0", 5],
        "Reinforce": [DATAROOT + SHAREPATH + "Reinforce/param_0/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_114", "C1", 1],
    }
    best_offline(pths, "Cont-CC-PermExDc-v0_Baseline", ylim=[-50, 0])
    """
    SHAREPATH = "output/test_v0/NonContexTT/Noncontext_PID_Alpha_Beta_Above_One_Buffer_Prefill/"
    pths = {
        "GAC w/ Entropy Beta Shift Buffer Prefill": [DATAROOT + SHAREPATH + "GAC/param_113/", "limegreen", 1],
        "GAC w/o Entropy Beta Shift Buffer Prefill": [DATAROOT + SHAREPATH + "GAC/param_257", "C1", 3],
    }
    best_offline(pths, "PID_GAC_Beta_Shift_Buffer_Prefill", ylim=[-10, 2])

def summary_plots():
    SHAREPATH = "output/test_v0/NonContexTT/Noncontext_PID_Alpha_Beta_Above_One_Buffer_Prefill/GAC"
    target_key = [
        "actor_info/param1",
        "proposal_info/param1",
        "actor_info/param2",
        "proposal_info/param2",
        "critic_info/Q",
        "env_info/constrain_detail/kp1",
        "env_info/constrain_detail/tau",
        "env_info/constrain_detail/height",
        "env_info/constrain_detail/flowrate",
        "env_info/constrain_detail/C1",
        "env_info/constrain_detail/C2",
        "env_info/constrain_detail/C3",
        "env_info/constrain_detail/C4",
        # "env_info/constrain",
        "env_info/lambda",
    ]
    pth = DATAROOT + SHAREPATH
    runs = os.listdir(pth)
    runs = [run for run in runs if run != ".DS_Store"]
    for run in runs:
        run_pth = os.path.join(pth, run)
        if os.path.isdir(run_pth):
            seeds = os.listdir(run_pth)
            seeds = [seed for seed in seeds if seed != ".DS_Store"]
            for seed in seeds:
                seed_pth = os.path.join(run_pth, seed)
                print(seed_pth)
                visualize_training_info(seed_pth, target_key, title="vis_noncontext_GAC_beta_shift_buffer_prefill_", threshold=0.99, xlim=None, ylim=[-2, 2])
    """
    run_pth = os.path.join(pth, "param_3")
    seed_pth = os.path.join(run_pth, "seed_0")
    visualize_training_info(seed_pth, target_key, title="vis_noncontext_GAC_alpha_beta_greater_than_one", threshold=0.99, xlim=None, ylim=[-2, 2])
    """


if __name__ == '__main__':
    sweep_parameter()
    #compare_algorithms()
    #summary_plots()


