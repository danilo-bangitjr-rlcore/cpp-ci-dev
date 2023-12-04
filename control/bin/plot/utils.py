import os
import json
import pickle
import numpy as np


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

def learning_curve_per_run(axs, data):
    for i, run in enumerate(data):
        p = axs[i].plot(run)


def sensitivity_curve(ax, data):
    params = list(data.keys())
    params.sort()
    sc = []
    stes = []
    for p in params:
        sc.append(data[p][:, -1].mean())
        stes.append(data[p][:, -1].std() / np.sqrt(len(data[p])) * 1.96)
    # ax.plot(sc)
    x = np.arange(len(params))
    ax.errorbar(x, sc, yerr=stes)
    print("Final performance", sc)
    print("STE", stes)
    print()
    ax.set_xticks(np.arange(len(params)), labels=params)


def param_sweep(ax, data):
    params = list(data.keys())
    print(params)
    params.sort()
    compare_auc = []
    compare_2nd_half = []
    compare_final = []
    for p in params:
        ax.plot(data[p].mean(axis=0), label=p)
        compare_auc.append(data[p].mean(axis=0).sum())
        compare_2nd_half.append(data[p][:, len(data[p])//2:].mean().sum())
        compare_final.append(data[p][:, -100:].mean().sum())
    
    best_auc_idx = np.array(compare_auc).argmax()
    print("Sweeping AUC:", params[best_auc_idx], compare_auc[best_auc_idx])
    best_2ndhalf_idx = np.array(compare_2nd_half).argmax()
    print("Sweeping the 2nd half:", params[best_2ndhalf_idx], compare_2nd_half[best_2ndhalf_idx])
    best_final_idx = np.array(compare_final).argmax()
    print("Sweeping Final:", params[best_final_idx], compare_final[best_final_idx])
    print("")


def load_param(pth, xlim=[], pick_seed=None):
    returns = []
    constraints = []
    runs = os.listdir(pth)
    if pick_seed is None:
        pick_seed = runs
    for r in runs:
        if r not in pick_seed:
            continue
        p = os.path.join(pth, r)
        ret = np.load(p + "/train_logs.npy")
        if os.path.isfile(p + "/ep_constraints.npy"):
            cons = np.load(p + "/ep_constraints.npy")
        else:
            cons = []
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
    perf_ret = {}
    perf_cons = {}
    for p in params:
        pth = os.path.join(root, p)
        setting, returns, constraints = load_param(pth)
        if sweep_param:
            add = True
            for tp, tv in fix_params.items():
                if setting[tp] != tv:
                    add = False
            if add:
                assert setting[sweep_param] not in perf_ret
                perf_ret[setting[sweep_param]] = returns
                perf_cons[setting[sweep_param]] = constraints
        else:
            if not np.isnan(returns.mean()):
                perf_ret[p] = returns
                perf_cons[p] = constraints
        # print(p, perf_ret[p].shape, perf_ret[p].mean(), perf_ret[p])

    plot_fn(axs[0], perf_ret)
    if len(axs) > 1:
        plot_fn(axs[1], perf_cons)


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
