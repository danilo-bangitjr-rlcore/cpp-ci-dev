import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt


def learning_curve(ax, data, label="", color=None, zorder=None):
    idx = np.arange(data.shape[1])
    avg = data.mean(axis=0)
    ste = data.std(axis=0) / np.sqrt(len(data))
    if color is None:
        p = ax.plot(idx, avg, label=label, zorder=zorder)
        color = p[0].get_color()
    else:
        p = ax.plot(idx, avg, label=label, color=color, zorder=zorder)
    ax.fill_between(idx, avg - ste * 1.96, avg + ste * 1.96, alpha=0.3, linewidth=0., color=color)
    return p

def learning_curve_per_run(axs, data):
    for i, run in enumerate(data):
        p = axs[i].plot(run)

#
def sensitivity_curve(ax, data, root):
    params = list(data.keys())
    params.sort()
    sc = []
    stes = []
    for p in params:
        sc.append(data[p].sum(axis=1).mean())
        stes.append(data[p].sum(axis=1).std() / np.sqrt(len(data[p])) * 1.96)
    # ax.plot(sc)
    x = np.arange(len(params))
    im = ax.errorbar(x, sc, yerr=stes)
    print("Final performance", sc)
    print("STE", stes)
    print()
    ax.set_xticks(np.arange(len(params)), labels=params)
    return im

def sensitivity_heatmap(ax, data, root):
    keys = np.array(list(data.keys()))
    params1 = list(set(keys[:, 0]))
    params2 = list(set(keys[:, 1]))
    params1.sort()
    params2.sort()
    sc = np.zeros((len(params1), len(params2)))
    for i1, p1 in enumerate(params1):
        for i2, p2 in enumerate(params2):
           sc[i1, i2] = data[(p1, p2)].sum(axis=1).mean()
    im = ax.imshow(sc)
    ax.set_yticks(np.arange(len(params1)), labels=params1)
    ax.set_xticks(np.arange(len(params2)), labels=params2)
    return im

def img_update_vim_vmax(ims, vmx=None):
    if vmx is None:
        data = []
        for im in ims:
            data.append(im.get_array())
        data = np.array(data)
        vmax, vmin = data.max(), data.min()
    else:
        vmin, vmax = vmx
    for im in ims:
        im.set_clim(vmin=vmin, vmax=vmax)

def param_sweep(ax, data, root):
    params = list(data.keys())
    params.sort()
    compare_auc = []
    compare_2nd_half = []
    compare_final = []
    for p in params:
        ax.plot(data[p].mean(axis=0), label=p)
        fig, p_ax = plt.subplots(figsize=(15, 10))
        p_ax.plot(data[p].mean(axis=0), label=p)
        p_ax.set_xlabel("Time Step")
        p_ax.set_ylabel("Return")
        p_ax.set_yscale('symlog', linthresh=0.1)
        fig.savefig(root + "/" + p + ".png")
        plt.close(fig)
        compare_auc.append(data[p].mean(axis=0).sum())
        compare_2nd_half.append(data[p][:, len(data[p])//2:].mean().sum())
        compare_final.append(data[p][:, -100:].mean().sum())
    
    sort_compare_auc = np.argsort(compare_auc)
    print("Best Performing Indices:")
    print(sort_compare_auc)
    print("AUCs:")
    print(compare_auc)
    print("2nd Half AUCs:")
    print(compare_2nd_half)
    print("Final 100 AUCs:")
    print(compare_final)
    best_auc_idx = np.array(compare_auc).argmax()
    print("Sweeping AUC:", params[best_auc_idx], compare_auc[best_auc_idx])
    best_2ndhalf_idx = np.array(compare_2nd_half).argmax()
    print("Sweeping the 2nd half:", params[best_2ndhalf_idx], compare_2nd_half[best_2ndhalf_idx])
    best_final_idx = np.array(compare_final).argmax()
    print("Sweeping Final:", params[best_final_idx], compare_final[best_final_idx])
    print("")


def load_logs(pth, pick_seed=None):
    """
    Loads logs for a single path and returns a list, one element for each seed
    """
    runs = os.listdir(pth)
    runs = [run for run in runs if run != ".DS_Store"]
    if pick_seed is None:
        pick_seed = runs
    
    param_list = []
    log_list = []
    
    for r in runs:
        if r not in pick_seed:
            continue
        p = os.path.join(pth, r)
        if os.path.isdir(p):
            with open(os.path.join(p, "info_logs.pkl"), "rb") as f:
                log = pickle.load(f)
            
            with open(os.path.join(p, "config.json"), "r") as f:
                params = json.load(f)
            
            log_list.append(log)
            param_list.append(params)
            
    return param_list, log_list


def load_param(pth, xlim=[], pick_seed=None):
    returns = []
    constraints = []
    runs = os.listdir(pth)
    runs = [run for run in runs if run != ".DS_Store"]
    if pick_seed is None:
        pick_seed = runs
    for r in runs:
        if r not in pick_seed:
            continue
        p = os.path.join(pth, r)
        if os.path.isdir(p):
            ret = np.load(p + "/train_logs.npy")
            if os.path.isfile(p + "/ep_constraints.npy"):
                cons = np.load(p + "/ep_constraints.npy")
            else:
                cons = []
                if os.path.isfile(p + "/info_logs.pkl"):
                    with open(p+"/info_logs.pkl", "rb") as f:
                        info = pickle.load(f)
                    for step in info:
                        try:
                            cons.append(step["env_info/constrain"])
                        except:
                            pass
                cons = np.array(cons)
                
            if xlim != []:
                ret = ret[xlim[0]: xlim[1]]
                cons = cons[xlim[0]: xlim[1]]
            returns.append(ret)
            constraints.append(cons)

    with open(os.path.join(pth, runs[0]) + "/config.json", "r") as f:
        params = json.load(f)
    return params, np.array(returns), np.array(constraints)


def load_exp(axs, plot_fn, root, fix_params={}, sweep_param=None):
    params = os.listdir(root)
    params = [param for param in params if param != ".DS_Store"]
    perf_ret = {}
    perf_cons = {}
    for p in params:
        pth = os.path.join(root, p)
        if os.path.isdir(pth):
            setting, returns, constraints = load_param(pth)
            if sweep_param:
                add = True
                for tp, tv in fix_params.items():
                    if setting[tp] != tv:
                        add = False
                if add:
                    if type(sweep_param) == str:
                        assert setting[sweep_param] not in perf_ret
                        perf_ret[setting[sweep_param]] = returns
                        perf_cons[setting[sweep_param]] = constraints
                    elif type(sweep_param) == list and len(sweep_param) == 2:
                        perf_ret[(setting[sweep_param[0]], setting[sweep_param[1]])] = returns
                        perf_cons[(setting[sweep_param[0]], setting[sweep_param[1]])] = constraints
            else:
                if not np.isnan(returns.mean()):
                    perf_ret[p] = returns
                    perf_cons[p] = constraints
            # print(p, perf_ret[p].shape, perf_ret[p].mean(), perf_ret[p])
    
    im = plot_fn(axs[0], perf_ret, root)
    if len(axs) > 1:
        plot_fn(axs[1], perf_cons)
    return im


def log2num(step_i, k_pth):
    if len(k_pth) == 1:
        return step_i[k_pth[0]]
    else:
        return log2num(step_i[k_pth[0]], k_pth[1:])

def log2ary(info, k):
    k_pth = k.split("/")
    ary = []
    for step_i in info:
        elem = log2num(step_i, k_pth)
        ary.append(elem)
    return np.array(ary)

import sys
def reduce_log_file_size(root):
    params = os.listdir(root)
    params = [param for param in params if param != ".DS_Store"]
    for p in params:
        pth = os.path.join(root, p)
        seeds = os.listdir(pth)
        for seed in seeds:
            info_file = os.path.join(pth, seed) + "/info_logs.pkl"
            with open(info_file, "rb") as f:
                info_log = pickle.load(f)
            for i_log in info_log:
                i_log.pop('action_visits', None)
                i_log['critic_info'].pop('Q-function', None)
            with open(info_file, "wb") as f:
                pickle.dump(info_log, f)