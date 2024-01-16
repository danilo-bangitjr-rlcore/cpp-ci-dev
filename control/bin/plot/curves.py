import json
import matplotlib.pyplot as plt
import itertools
from textwrap import wrap
from utils import *

DATAROOT = "../../out/"


def sweep_offline(pth_base, agent="GAC"):
    print(agent)
    root = pth_base.format(agent)
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    load_exp([axs], param_sweep, root, {}, None)
    plt.legend()
    axs.set_ylabel("Performance")
    axs.set_xlabel("Episodes")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/sweep_{}.png".format(agent), dpi=300, bbox_inches='tight')

def sensitivity_plot(pth_base, agent, fix_params_choices, sweep_param, title):
    root = pth_base.format(agent)
    fig, axs = plt.subplots(1, len(fix_params_choices), figsize=(5*len(fix_params_choices), 4))
    for i, fix_params in enumerate(fix_params_choices):
        load_exp([axs[i]], sensitivity_curve, root, fix_params, sweep_param)
    plt.legend()
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')

def sensitivity_plot_2d(pth_base, agent, fix_params_choices, sweep_param1, sweep_param2, title):
    font = {'size': 7}
    plt.rc('font', **font)
    root = pth_base.format(agent)
    fig, axs = plt.subplots(1, len(fix_params_choices), figsize=(5*len(fix_params_choices), 4))
    axs = [axs] if len(fix_params_choices) == 1 else axs
    imgs = []
    for i, fix_params in enumerate(fix_params_choices):
        im = load_exp([axs[i]], sensitivity_heatmap, root, fix_params, [sweep_param1, sweep_param2])
        imgs.append(im)
        axs[i].set_title(fix_params)
        axs[i].set_title(fix_params)
    img_update_vim_vmax(imgs, vmx=[-10000, 6000])

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.05, 0.8])
    fig.colorbar(imgs[-1], cax=cbar_ax)
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')


def best_offline(pths, title, ylim):
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    # axins = zoomed_inset_axes(axs, 6, loc=1)
    axs = [axs]
    for label, [pth, c, z] in pths.items():
        setting, returns, constraints = load_param(pth)
        learning_curve(axs[0], returns, label=label, color=c, zorder=z)
    axs[0].set_ylim(ylim)
    axs[0].legend()
    # axs[1].legend()
    axs[0].set_ylabel("Performance")
    # axs[1].set_ylabel("Constraint")
    # axs[1].set_xlabel("Episodes")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')

def best_offline_per_run(pth, title):
    setting, returns, constraints = load_param(pth)
    fig, axs = plt.subplots(1, len(returns), figsize=(3*len(returns), 3))
    if len(returns) == 1:
        axs = [axs]
    learning_curve_per_run(axs, returns)
    # axs[0].set_ylim(-10, 2)
    axs[0].legend()
    axs[0].set_ylabel("Performance")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')


def reproduce_demo(pths, title, ylim):
    def recover_paper_data(returns):
        # denormalization
        returns = returns * 8 - 8
        # smooth
        smoothed = np.zeros(returns.shape)
        smoothed[:, :5] = returns[:, :5]
        for step in range(5, returns.shape[1]):
            smoothed[:, step] = returns[:, step]
            smoothed[:, step] = smoothed[:, step - 4: step + 1].mean()
        return smoothed
    # fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    axs = [axs]
    for label, [pth, c, z] in pths.items():
        setting, returns, constraints = load_param(pth)#, pick_seed=["seed_0", "seed_42"])
        returns = recover_paper_data(returns)
        learning_curve(axs[0], returns, label=label, color=c, zorder=z)
        # constraints = recover_paper_data(constraints)
        # learning_curve(axs[1], constraints, label=label, color=c, zorder=z)
    axs[0].set_ylim(ylim)
    axs[0].legend()
    # axs[1].legend()
    axs[0].set_ylabel("Performance")
    # axs[1].set_ylabel("Constraint")
    # axs[1].set_xlabel("Episodes")
    fig.tight_layout()
    plt.savefig(DATAROOT + "img/{}.png".format(title), dpi=300, bbox_inches='tight')

def visualize_training_info(target_file, target_key, title='vis_training', threshold=None, xlim=None, ylim=None):
    with open(target_file+"/info_logs.pkl", "rb") as f:
        info = pickle.load(f)
    ret = np.load(target_file+"/train_logs.npy")
    
    reformat = {}
    for k in target_key:
        ary = log2ary(info, k)
        if len(ary.shape) > 2 and ary.shape[2] == 1:
            ary = ary.squeeze(2)
        if len(ary.shape) > 1 and ary.shape[1] == 1:
            ary = ary.squeeze(1)
        reformat[k] = ary

    fig = plt.figure(layout='constrained', figsize=(12, 3*len(target_key)))
    subfigs = fig.subfigures(len(target_key)+1, 1, wspace=0.07)

    ax = subfigs[0].subplots(1, 1)
    ax.plot(ret)
    ax.set_title("Reward")
    axes = [ax]
    for i, k in enumerate(target_key):
        multidim = len(reformat[k].shape)
        dim = 1 if multidim == 1 else reformat[k].shape[1]
        axs = subfigs[i+1].subplots(1, dim, sharey=False)
        if dim == 1:
            axes.append(axs)
            axs.plot(reformat[k])
            axs.set_title(k)
            print(k, reformat[k][-10:].mean(), reformat[k][-10:].std(),  reformat[k][-10:].max())
           
        else:
            for d in range(dim):
                axes.append(axs[d])
                axs[d].plot(reformat[k][:, d])
                axs[d].set_title(k+"/dimension-{}".format(d))
                print(k, d, reformat[k][:, d][-10:].mean(), reformat[k][:, d][-10:].std())

    highlight = []
    if threshold is not None:
        highlight = np.where(ret < threshold)[0]
        for ax in axes:
            for x in highlight:
                ax.axvline(x, linestyle='-', color='lightgrey', linewidth=1, zorder=-1)
        print("Timesteps where reward is lower than threshold, after 2500: \n", highlight[np.where(highlight>2500)])

    range_ = ""
    if xlim is not None:
        range_ = "_{}-{}".format(xlim[0], xlim[1])
        for ax in axes:
            ax.set_xlim(xlim)

    if ylim is not None:
        axes[0].set_ylim(ylim)

    # fig, axs = plt.subplots(len(target_key)+1, 1, figsize=(12, 3*len(target_key)))
    # axs[0].plot(ret)
    # axs[0].set_title("Reward")
    # for i, k in enumerate(target_key):
    #     print(reformat[k].shape)
    #     axs[i + 1].plot(reformat[k])
    #     axs[i + 1].set_title(k)

    # highlight = []
    # if threshold is not None:
    #     highlight = np.where(ret < threshold)[0]
    #     for ax in axs:
    #         for x in highlight:
    #             ax.axvline(x, linestyle='-', color='lightgrey', linewidth=1, zorder=-1)
    #     print("Timesteps where reward is lower than threshold, after 2500: \n", highlight[np.where(highlight>2500)])
    #
    # range_ = ""
    # if xlim is not None:
    #     range_ = "_{}-{}".format(xlim[0], xlim[1])
    #     for ax in axs:
    #         ax.set_xlim(xlim)
    #
    # if ylim is not None:
    #     axs[0].set_ylim(ylim)

    # fig.tight_layout()
    config_f = open(target_file + "/config.json")
    config = json.load(config_f)

    plt.savefig(target_file + "/{}{}_replay_{}_tau_{}_rho_{}_prop_rho_mult_{}_actor_lr_{}_critic_lr_{}.png".format(title, range_, config["buffer_size"], config["tau"], config["rho"], config["prop_rho_mult"], config["lr_actor"], config["lr_critic"]), dpi=300, bbox_inches='tight')
    plt.savefig(DATAROOT + "img/{}{}_replay_{}_tau_{}_rho_{}_prop_rho_mult_{}_actor_lr_{}_critic_lr_{}.png".format(title, range_, config["buffer_size"], config["tau"], config["rho"], config["prop_rho_mult"], config["lr_actor"], config["lr_critic"]), dpi=300, bbox_inches='tight')
    plt.close()
    # fig.close()
    return highlight

def draw_q_functions(pth_base, fixed_params_list, agent, itr=-1, num_rows=1):
    keys, values = zip(*fixed_params_list.items())
    fix_params_choices = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    font = {'size': 7}
    plt.rc('font', **font)
    root = pth_base.format(agent)
    
    num_cols = len(fix_params_choices)//num_rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
    axs = [axs] if len(fix_params_choices) == 1 else axs    

    params = os.listdir(root)
    params = [param for param in params if param != ".DS_Store"]
    ims = []
    for p in params:
        param_list, log_list = load_logs(root +'/' + p)
        setting = param_list[0] # all seeds have same params
        
        for fixed_params_index, fix_params_choice in enumerate(fix_params_choices):
            add = True
            for tp, tv in fix_params_choice.items():
                if setting[tp] != tv:
                    add = False        
            if add == True:
                break
    
        if add == True:
            for log in log_list:
                q = log[itr]['critic_info']['Q-function'][0]
                row = fixed_params_index//num_cols
                col = fixed_params_index%num_cols
                if num_rows != 1:
                    im = axs[row, col].imshow(q,  vmin=-1, vmax=2)
                    axs[row, col].set_title("\n".join(wrap(str(fix_params_choice), 60)))
                else:
                    im = axs[col].imshow(q)
                    axs[col].set_title(fix_params_choice, 60)
                    
                ims.append(im)
                    
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.95, 0.7, 0.05, 0.2])
    fig.colorbar(ims[-1], cax=cbar_ax)
    
    return axs