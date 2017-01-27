import numpy as np
import pickle
import pandas as pd
import os.path as op

from nistats.design_matrix import make_design_matrix, plot_design_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(efficiencies, contrast, i_best=-1, perc_best=0, i_worst=-1, perc_worst=0, eff_min=-1, eff_max=-1):
    hist = np.histogram(efficiencies, bins=30)
    height = 1.1 * max(hist[0])
    width = hist[1][1] - hist[1][0]
    plt.bar(hist[1][:-1], hist[0], width=width)

    if i_best != -1:
        plt.plot([efficiencies[i_best], efficiencies[i_best]], [0, height], '-m', linewidth=2)
        plt.text(efficiencies[i_best] + width, 0.8 * height, "%.1f%%\n%f" %
                 (perc_best * 100, efficiencies[i_best]), color='m', size=12, weight='bold')

    if i_worst != -1:
        plt.plot([efficiencies[i_worst], efficiencies[i_worst]], [0, height], '-r', linewidth=2)
        plt.text(efficiencies[i_worst] + width, 0.1 * height, "%.1f%%\n%f" %
                 (perc_worst * 100, efficiencies[i_worst]), color='r', size=12, weight='bold')

    if eff_min != -1 and eff_max != -1:
        plt.xlim((eff_min, eff_max))

    plt.ylim((0, height))
    plt.title(contrast)


def show_distribs(params_file, designs_file, efficiencies_file):
    params = pickle.load(open(params_file, 'rb'))
    contrasts = params['contrasts']

    nbr_designs = 1000
    nbr_tirages = 6
    dsgns, eff = split_dataset(designs_file, efficiencies_file, nbr_designs, nbr_tirages)

    plt.figure(figsize=(18, 10))
    n_rows = 2
    n_cols = nbr_tirages / n_rows
    i_cont = 0
    # i_tirage = 0
    for i_tirage in range(nbr_tirages):
        i_best, percs_best = find_best_design(eff[i_tirage], contrasts)
        i_worst, percs_worst = find_worst_design(eff[i_tirage], contrasts)

        plt.subplot(n_rows, n_cols, i_tirage + 1)
        plot_distribution(eff[i_tirage][i_cont], contrasts[i_cont],
                          i_best, percs_best[i_cont], i_worst, percs_worst[i_cont])


def load_results(result_files, legends):
    eff_vect = []
    k_vect = []
    const_vect = []
    for i, file in enumerate(result_files):
        data_file = pickle.load(open(file, "rb"))
        data_tmp = data_file['data']
        eff_vect.append(data_tmp[:, 1])
        k_vect.append(data_tmp[:, 0])
        const_vect.append(np.repeat(legends[i], data_tmp.shape[0]))
    eff_vect = np.array(eff_vect).flatten()
    const_vect = np.array(const_vect).flatten()
    k_vect = np.array(k_vect).flatten()
    data = {'Constraints': const_vect, 'Number of designs': k_vect, 'Efficiency': eff_vect}
    return data


def plot_points(result_files, legends, fig_file):
    data = []
    nbr_designs = {}
    for i, file in enumerate(result_files):
        data_file = pickle.load(open(file, "rb"))
        data.append(data_file['data'])
    data = np.array(data)
    # ymin = min([np.min(data[:, 1]), np.min(data[:, 3])])
    # ymax = max([np.max(data[:, 1]), np.max(data[:, 3])])
    ymin = np.min(data[:, :, 1])
    ymax = np.max(data[:, :, 1])

    fig = plt.figure(figsize=(14,10))
    colors = ['r', 'g', 'b']
    for i, file in enumerate(result_files):
        plt.scatter(data[i, :, 0], data[i, :, 1], c=colors[i], alpha=0.8, s=60)

    plt.ylim((0.95*ymin, 1.05*ymax))
    plt.xlabel("Number of designs")
    plt.ylabel("Efficiency")
    plt.xscale('log')
    plt.legend(legends)

    # fig.savefig(fig_file)


def plot_box(result_files, legends, fig_file):
    data = load_results(result_files, legends)

    fig = plt.figure(figsize=(18, 9))
    sns.boxplot(x='Number of designs', y="Efficiency", hue="Constraints", data=data, palette="Set2")
    plt.xlabel("Number of designs", fontsize=14)
    plt.ylabel("Efficiency", fontsize=14)
    plt.title("Best efficiency vs. Number of designs used to find best design", fontsize=16)


def find_best_in_all(result_file):
    best = {}
    eff_max = 0
    data_file = pickle.load(open(result_file, "rb"))
    data_tmp = data_file['data']
    for j in range(len(data_tmp[:, 1])):
        eff = data_tmp[j, 1]
        if eff > eff_max:
            best['efficiency'] = eff
            best['i_best'] = j
            best['nbr_designs'] = data_tmp[j, 0]
            best['file'] = file
            eff_max = eff
    return best


def plot_design(designs_file, design_index, params_file):
    params = pickle.load(open(params_file, 'rb'))
    tr = params['tr']
    durations = params['cond_durations']

    designs = np.load(designs_file)
    design = designs[:, design_index]
    conditions = design[1]
    onsets = design[0]
    total_duration = onsets[-1] + durations[int(conditions[-1])] + params['SOAmax']
    n_scans = np.ceil(total_duration/tr)
    frame_times = np.arange(n_scans) * tr

    paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets})
    X = make_design_matrix(frame_times, paradigm, drift_model='blank')
    plot_design_matrix(X)


def plot_distribs(params_file, efficiencies_file, design_index, fig_file=None):
    params = pickle.load(open(params_file, 'rb'))
    contrasts = params['contrasts']
    i_contrast = 0

    efficiencies = np.load(efficiencies_file)

    # i_sec = 600#400
    # avgs = np.mean(efficiencies, axis=1)
    # effs_inf = 0.8*avgs
    # effs_sup = 0.98*avgs
    # for i in range(efficiencies.shape[1]):
    #     effs = efficiencies[:, i]
    #     if sum(effs > effs_inf) == 9 and sum(effs < effs_sup) == 9:
    #         i_sec = i
    #         break

    print("Best design index: {}".format(design_index))
    # print("Second choosen design indexe: {}".format(i_sec))

    n_rows = 3
    n_cols = int(np.ceil(contrasts.shape[0] / n_rows))

    fig = plt.figure(figsize=(15,10))
    for i, c in enumerate(contrasts):
        hist = np.histogram(efficiencies[i], bins=30)
        best_eff = efficiencies[i, design_index]
        inf_indexes = efficiencies[i] <= best_eff
        eff_rep = np.sum(efficiencies[i, inf_indexes]) / np.sum(efficiencies[i], dtype=float)

        # secondary design
        # sec_eff = efficiencies[i, i_sec]
        # inf_indexes = efficiencies[i] <= sec_eff
        # eff_rep_sec = np.sum(efficiencies[i, inf_indexes]) / np.sum(efficiencies[i], dtype=float)

        height = 1.1 * max(hist[0])
        width = hist[1][1] - hist[1][0]

        plt.subplot(n_rows, n_cols, i + 1)
        plt.bar(hist[1][:-1], hist[0], width=width)
        plt.plot([best_eff, best_eff], [0, height], '-m', linewidth=2)
        plt.text(best_eff + width, 0.8 * height, "{:.4f}\n{:.1%}".format(best_eff, eff_rep),
                 color='m', size=12, weight='bold')

        # plt.plot([sec_eff, sec_eff], [0, height], '-r', linewidth=2)
        # plt.text(sec_eff + width, 0.1 * height, "%.1f%%\n%f" %
        #          (eff_rep_sec * 100, sec_eff), color='r', size=12, weight='bold')

        # plt.xlim((0, np.max(efficiencies)*1.1))
        plt.ylim((0, height))
        plt.title(c)

    if fig_file is not None:
        fig.savefig(fig_file)


def plot_simple_design(ax, design, tr, durations, SOAmax):
    conditions = design[1]
    onsets = design[0]
    print("WARNING: DESIGN DURATION HAS CHANGED")
    # total_duration = onsets[-1] + durations[int(conditions[-1])] + SOAmax
    total_duration = onsets[-1] + float(durations[0][-1].replace(",", "."))+ SOAmax

    n_scans = np.ceil(total_duration/tr)
    frame_times = np.arange(n_scans) * tr

    # pool condtions
    # print("WARNING CONDITIONS ARE POOLED")
    # conditions = np.array(np.mod(conditions,12), dtype=int)
    # conditions = np.array(conditions/12, dtype=int)

    paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets})
    X = make_design_matrix(frame_times, paradigm, drift_model='blank')
    plot_design_matrix(X, ax=ax)


# def plot_n_best_matrix(designs_file, params_file, efficiencies_file, design_indexes, fig_file=None):
#     # 0 - Loads data and design's parameters
#     params = pickle.load(open(params_file, 'rb'))
#     tr = params['tr']
#     durations = params['cond_durations']
#     SOAmax = params['SOAmax']
#     contrasts = params['contrasts']
#
#     designs = np.load(designs_file)
#     efficiencies = np.load(efficiencies_file)
#
#     # Plot nbr_matrix first designs matrixes
#     fig = plt.figure(figsize=(21, 13))
#     n_rows = int(np.ceil(np.sqrt(len(design_indexes))))
#     n_cols = int(np.ceil(len(design_indexes)/float(n_rows)))
#     for i, index in enumerate(design_indexes):
#         design = designs[:, index]
#         ax = plt.subplot(n_rows, n_cols, i+1)
#         plot_simple_design(ax, design, tr, durations, SOAmax)
#         plt.title("{} - design n°{}".format(i+1, index))
#         # for j, c in enumerate(contrasts):
#         #     plt.text(5.6, 100 + j*50, "{:.5f}".format(efficiencies[j, index]), color='m', size=12)
#
#     if fig_file is not None:
#         fig.savefig(fig_file)

def plot_n_best_matrix(designs_file, params_file, efficiencies_file, design_indexes, fig_file=None):
    # 0 - Loads data and design's parameters
    params = pickle.load(open(params_file, 'rb'))
    tr = params['tr']
    # durations = params['cond_durations']
    SOAmax = params['SOAmax']
    contrasts = params['contrasts']

    designs = np.load(designs_file)
    efficiencies = np.load(efficiencies_file)

    durations_tab = np.load("/hpc/banco/bastien.c/data/optim/VC/part10A/durations_tab.npy")

    # Plot nbr_matrix first designs matrixes
    fig = plt.figure(figsize=(21, 13))
    n_rows = int(np.ceil(np.sqrt(len(design_indexes))))
    n_cols = int(np.ceil(len(design_indexes)/float(n_rows)))
    for i, index in enumerate(design_indexes):
        durations = durations_tab[index]
        design = designs[:, index]
        ax = plt.subplot(n_rows, n_cols, i+1)
        plot_simple_design(ax, design, tr, durations, SOAmax)
        plt.title("{} - design n°{}".format(i+1, index))
        # for j, c in enumerate(contrasts):
        #     plt.text(5.6, 100 + j*50, "{:.5f}".format(efficiencies[j, index]), color='m', size=12)

    if fig_file is not None:
        fig.savefig(fig_file)

if __name__ == "__main__":
    # PRINT BEST EFFICIENCIES VS NBR OF DESIGNS
    # legend = ['No constraint', 'Alpha 95%']
    # files = [result_path + 'data_no_constraint.p', result_path + "data_a95.p"]
    # plot_points(files, legend, result_path + fig_name)
    # plot_box(files, legend, result_path + fig_name)

    # # Find best design
    # best = find_best_in_all([files[1]], [legend[1]])
    # print("Best design obtained with following charasteristics:")
    # print(best)
    #
    # # Plot design
    # if best['file'] == '/hpc/banco/bastien.c/data/optim/VL/test_k/data_no_constraint.p':
    #     designs_file = '/hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/designs.npy'
    #     params_file = '/hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/params.p'
    #     eff_file = '/hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/efficiencies.npy'
    # elif best['file'] == '/hpc/banco/bastien.c/data/optim/VL/test_k/data_a95.p':
    #     designs_file = '/hpc/banco/bastien.c/data/optim/VL/banks/alpha_constraint/95/designs.npy'
    #     params_file = '/hpc/banco/bastien.c/data/optim/VL/banks/alpha_constraint/95/params.p'
    #     eff_file = '/hpc/banco/bastien.c/data/optim/VL/banks/alpha_constraint/95/efficiencies.npy'

    # PLOT N BEST DESGIN MATRIXES
    path = "/hpc/banco/bastien.c/data/optim/VC/iti_var/"
    best_indexes = np.load(op.join(path, "bests.npy"))
    plot_n_best_matrix(op.join(path, "designs.npy"), op.join(path, "params.p"), op.join(path, "efficiencies.npy"),
                       best_indexes, op.join(path, "design_matrix_bests.png"))

    for index in best_indexes:
        plot_distribs(op.join(path, "params.p"), op.join(path, "efficiencies.npy"), index,
                      op.join(path, "distribs_bests/{}.png".format(index)))
    #
    # best_indexes = np.load(op.join(path, "moys.npy"))
    # plot_n_best_matrix(op.join(path, "designs.npy"), op.join(path, "params.p"), op.join(path, "efficiencies.npy"),
    #                    best_indexes, op.join(path, "design_matrix_moyss.png"))
    #
    # for index in best_indexes:
    #     plot_distribs(op.join(path, "params.p"), op.join(path, "efficiencies.npy"), index,
    #                   op.join(path, "distribs_moys/{}.png".format(index)))


    # plot_design("/hpc/banco/bastien.c/data/optim/VC/part02A/designs.npy", 562,
    #             "/hpc/banco/bastien.c/data/optim/VC/part02A/params.p")
    # plt.show()
