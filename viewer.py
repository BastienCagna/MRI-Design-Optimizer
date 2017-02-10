# Because otherwise "$DISPLAY" is not defined in batch mode
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pickle
import pandas as pd
import os.path as op
import sys
from os import system

from nistats.design_matrix import make_design_matrix, plot_design_matrix
from design_efficiency import design_matrix

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


def plot_points(result_files, legends, fig_file=None):
    data = []
    for i, file in enumerate(result_files):
        data_file = pickle.load(open(file, "rb"))
        data.append(data_file['data'])
    data = np.array(data)
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

    if fig_file is not None:
        fig.savefig(fig_file)


def plot_box(result_files, legends, fig_file=None):
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

    fig = plt.figure(figsize=(18, 9))
    sns.boxplot(x='Number of designs', y="Efficiency", hue="Constraints", data=data, palette="Set2")
    plt.xlabel("Number of designs", fontsize=14)
    plt.ylabel("Efficiency", fontsize=14)
    plt.title("Best efficiency vs. Number of designs used to find best design", fontsize=16)

    if fig_file is not None:
        fig.savefig(fig_file)


def plot_distribs(design, efficiencies, fig_file=None):
    params = pickle.load(open(params_file, 'rb'))
    contrasts = params['contrasts']

    efficiencies = np.load(efficiencies_file)

    # TODO: remove fixed rows count
    n_rows = 3
    n_cols = int(np.ceil(contrasts.shape[0] / n_rows))

    fig = plt.figure(figsize=(15,10))
    for i, c in enumerate(contrasts):
        hist = np.histogram(efficiencies[i], bins=30)
        best_eff = efficiencies[i, design_index]
        inf_indexes = efficiencies[i] <= best_eff
        eff_rep = np.sum(efficiencies[i, inf_indexes]) / np.sum(efficiencies[i], dtype=float)

        height = 1.1 * max(hist[0])
        width = hist[1][1] - hist[1][0]

        plt.subplot(n_rows, n_cols, i + 1)
        plt.bar(hist[1][:-1], hist[0], width=width)
        plt.plot([best_eff, best_eff], [0, height], '-m', linewidth=2)
        plt.text(best_eff + width, 0.8 * height, "{:.4f}\n{:.1%}".format(best_eff, eff_rep),
                 color='m', size=12, weight='bold')

        plt.ylim((0, height))
        plt.title(c)

    if fig_file is not None:
        fig.savefig(fig_file)


# TODO: add optional conditions grouping parameter
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


def plot_n_matrix(designs_indexes, designs, tr, fig_file=None):
    # Plot nbr_matrix first designs matrixes
    fig = plt.figure(figsize=(21, 13))

    n_rows = int(np.ceil(np.sqrt(len(designs_indexes))))
    n_cols = int(np.ceil(len(designs_indexes)/float(n_rows)))

    for i, idx in enumerate(designs_indexes):
        ax = plt.subplot(n_rows, n_cols, i+1)
        x = design_matrix(tr, designs[idx])
        plot_design_matrix(x, ax=ax)
        plt.title("{} - design n°{}".format(i+1, idx))

    if fig_file is not None:
        fig.savefig(fig_file)


if __name__ == "__main__":
    # PRINT BEST EFFICIENCIES VS NBR OF DESIGNS
    # legend = ['No constraint', 'Alpha 95%']
    # files = [result_path + 'data_no_constraint.p', result_path + "data_a95.p"]
    # plot_points(files, legend, result_path + fig_name)
    # plot_box(files, legend, result_path + fig_name)

    # PLOT N BEST DESGIN MATRIXES
    param_path = sys.argv[1]
    if len(sys.argv) > 2:
        params = pickle.load(open(op.join(param_path, sys.argv[2]), "rb"))
    else:
        params = pickle.load(open(op.join(param_path, "params.p"), "rb"))
    path = params['output_path']

    best_indexes = np.load(op.join(path, "bests.npy"))
    plot_n_matrix(best_indexes, path, fig_file=op.join(path, "desgins_matrix_bests.png"))

    if not op.isdir(op.join(path, "distribs_bests")):
        system("mkdir {}".format(op.join(path, "distribs_bests")))
    for index in best_indexes:
        plot_distribs(op.join(path, "params.p"), op.join(path, "efficiencies.npy"), index,
                      op.join(path, "distribs_bests/{:05d}.png".format(index)))

    best_indexes = np.load(op.join(path, "avgs.npy"))
    plot_n_matrix(best_indexes, path, fig_file=op.join(path, "desgins_matrix_avgs.png"))

    if not op.isdir(op.join(path, "distribs_avgs")):
        system("mkdir {}".format(op.join(path, "distribs_avgs")))
    for index in best_indexes:
        plot_distribs(op.join(path, "params.p"), op.join(path, "efficiencies.npy"), index,
                      op.join(path, "distribs_avgs/{:05d}.png".format(index)))

