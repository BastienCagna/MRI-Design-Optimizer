import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nistats.design_matrix import plot_design_matrix

from design_optimisation.design_efficiency import design_matrix


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


def plot_distribs(design_idx, efficiencies, contrasts_names, fig_file=None):
    # n_rows = 3
    # n_cols = int(np.ceil(contrasts_names.shape[0] / n_rows))
    n_rows = int(np.ceil(np.sqrt(len(contrasts_names))))
    n_cols = int(np.ceil(len(contrasts_names)/float(n_rows)))

    fig = plt.figure(figsize=(15,10))
    for i, c in enumerate(contrasts_names):
        hist = np.histogram(efficiencies[i], bins=30)
        best_eff = efficiencies[i, design_idx]
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


def plot_n_matrix(designs_indexes, designs, tr, fig_file=None):
    # Plot nbr_matrix first designs matrixes
    fig = plt.figure(figsize=(21, 13))

    n_rows = int(np.ceil(np.sqrt(len(designs_indexes))))
    n_cols = int(np.ceil(len(designs_indexes)/float(n_rows)))

    for i, idx in enumerate(designs_indexes):
        ax = plt.subplot(n_rows, n_cols, i+1)
        x = design_matrix(designs[idx], tr)
        plot_design_matrix(x, ax=ax)
        plt.title("{} - design n°{}".format(i+1, idx))

    if fig_file is not None:
        fig.savefig(fig_file)


if __name__ == "__main__":
    params_file = sys.argv[1]
    design_list_file = sys.argv[2]
    design_idx = int(sys.argv[3])

    params = pickle.load(open(params_file, "rb"))
    data = pickle.load(open(design_list_file, "rb"))

    x_mat = design_matrix(data['designs'][design_idx], params['tr'])
    plot_design_matrix(x_mat)

    plt.show()
