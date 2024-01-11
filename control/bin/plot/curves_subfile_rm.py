import itertools
import os
import matplotlib.pyplot as plt


from curves import DATAROOT, sweep_offline, reproduce_demo, best_offline, sensitivity_plot_2d
from curves import visualize_training_info, draw_q_functions


def draw_q_plots():
    fixed_params_list = {
        "lr_critic": [10**i for i in range(-2, -6, -1)],
        "batch_size": [8, 64],
        "optimizer": ['RMSprop'],
        "etc_learning_start" : [0]  
    }
    pth_base = DATAROOT + "output/test_v0/NonContexTT/etc_criticetc_critic/{}"
    agent = 'ETC'

    draw_q_functions(pth_base, fixed_params_list, agent, itr=-1, num_rows=4)
    plt.savefig('q.png', bbox_inches='tight')


def draw_sensitivity_2d_buffer():
    fixed_params_list = {
        "tau": [2, 1 ,1e-1],
        "batch_size": [8, 32],
        "buffer_prefill": [100, 1000]
    }
    
    pth_base = DATAROOT + "output/test_v0/NonContexTT/09-01-2024-buffer_prefill"
    keys, values = zip(*fixed_params_list.items())
    fix_params_choices = [dict(zip(keys, v)) for v in itertools.product(*values)]
   
    sensitivity_plot_2d(pth_base+"/{}/".format('GAC'), 'GAC', fix_params_choices, 'lr_actor', 'lr_critic', 'Sensitivity')
    
def draw_sensitivity_2d_adam_RMSprop():
    fixed_params_list = {
        "optimizer": ['Adam', 'RMSprop']
    }
    
    pth_base = DATAROOT + "output/test_v0/NonContexTT/learning_rate_sweep_adam_RMS_prop/target0/replay5000_batch8/env_scale_10/"
    keys, values = zip(*fixed_params_list.items())
    fix_params_choices = [dict(zip(keys, v)) for v in itertools.product(*values)]
   
    sensitivity_plot_2d(pth_base+"/{}/".format('GAC'), 'GAC', fix_params_choices, 'lr_actor', 'lr_critic', 'Sensitivity')
    
    
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
    SHAREPATH = "output/test_v0/NonContexTT/Noncontext_PID_Baseline/"
    pths = {
        "ETC": [DATAROOT + SHAREPATH + "ETC/param_0/", "limegreen", 3],
        "GAC": [DATAROOT + SHAREPATH + "GAC/param_28", "C1", 1],
    }
    best_offline(pths, "PID_Baseline", ylim=[-10, 2])


def test():
    # file = DATAROOT + "output/test_v0/NonContexTT/09-01-2024-buffer_prefill/GAC"
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
    
    
    pth = DATAROOT + "output/test_v0/NonContexTT/09-01-2024-buffer_prefill/GAC"
    runs = os.listdir(pth)
    runs = [run for run in runs if run != ".DS_Store"]
    for run in runs:
        run_pth = os.path.join(pth, run)
        # visualize_training_info(run_pth, target_key, title="vis_noncontext_GAC", threshold=0.99, xlim=None, ylim=[-2, 2])
               
               
    
        if os.path.isdir(run_pth):
            seeds = os.listdir(run_pth)
            seeds = [seed for seed in seeds if seed != ".DS_Store"]
            for seed in seeds:
                seed_pth = os.path.join(run_pth, seed)
                print(seed_pth)
                visualize_training_info(seed_pth, target_key, title="vis_noncontext_GAC", threshold=0.99, xlim=None, ylim=[-2, 2])




if __name__ == '__main__':
    # test()
    # draw_sensitivity_2d()
    # sweep_parameter()
    # compare_algorithms()
    # draw_sensitivity_2d_buffer()
    draw_q_plots()


