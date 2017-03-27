import pickle
import sys
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nistats.design_matrix import plot_design_matrix

from design_optimisation.design_efficiency import design_matrix


def plot_distribution(efficiencies, contrast_name, i_best=-1, perc_best=0, i_worst=-1, perc_worst=0, eff_min=-1, eff_max=-1):
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
    plt.title(contrast_name)


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


def plot_n_matrix(designs_indexes, designs, tr, group_cond=False, fig_file=None):
    # Plot nbr_matrix first designs matrixes
    fig = plt.figure(figsize=(21, 13))

    n_rows = int(np.ceil(np.sqrt(len(designs_indexes))))
    n_cols = int(np.ceil(len(designs_indexes)/float(n_rows)))

    for i, idx in enumerate(designs_indexes):
        ax = plt.subplot(n_rows, n_cols, i+1)
        plot_design(designs[idx], tr, group=group_cond, ax=ax,
                    title="{} - design n°{}".format(i+1, idx))

    if fig_file is not None:
        fig.savefig(fig_file)


def plot_design(design, tr, title="Design matrix", group=False, ax=None):
    if group is True:
        design['trial_type'] = design['trial_group']
    x = design_matrix(design, tr)
    plot_design_matrix(x, ax=ax)
    plt.title(title)


def plot_histo(cond_count, names, groups=None, cond_names=None):
    if groups is not None:
        # Event count for each group of conditions
        counts = []
        for grp in np.unique(groups):
            # Each row of conds_of_grp will contains the list of conditions names associated to
            # this groups index
            cond_idx = np.argwhere(np.array(groups) == grp).flatten()
            cond_names_in_grp = []
            for c_idx in cond_idx:
                cond_names_in_grp.append(cond_names[c_idx])

            # Sum every conditions count for each group
            sum = 0
            for i, cond_grp in enumerate(groups):
                if cond_grp == grp:
                    sum += cond_count[cond_names[i]]
            counts.append(sum)
    else:
        counts = cond_count

    ind = np.arange(len(counts))
    plt.bar(ind, counts)
    plt.xticks(ind+0.4, names)
    plt.title("Condition counts")


if __name__ == "__main__":
    dir = "/hpc/banco/bastien.c/data/optim/calibrator_iti/designs_iti_0_0/"
    params = pickle.load(open(op.join(dir, "params.pck"), "rb"))
    work_dir = params['work_dir']

    print(op.join(work_dir, "efficiencies.npy"))
    data = pickle.load(open(op.join(work_dir, "designs.pck"), "rb"))
    designs = data['designs']
    efficiencies = np.load(op.join(work_dir, "efficiencies.npy"))

    # for i_c, contrast_name in enumerate(params['contrasts_def'].keys()):
    #     plt.figure()
    #     plot_distribution(efficiencies[i_c],  contrast_name, i_best=idx)
    # print(designs[92632]['trial_type'][-10:])
    # print(designs[92632]['trial_group'][-10:])
    contrast_name = params['contrasts_names'][0]
    plt.figure()
    ax = plt.subplot(2, 2, 1)
    plot_design(designs[48023], params['tr'], title="Average design", ax=ax)
    ax = plt.subplot(2, 2, 2)
    plot_design(designs[92632], params['tr'], title="Best design", ax=ax)
    ax = plt.subplot(2, 2, 3)
    plot_distribution(efficiencies[0],  contrast_name, i_best=48023)
    ax = plt.subplot(2, 2, 4)
    plot_distribution(efficiencies[0],  contrast_name, i_best=92632)
    plt.tight_layout()

    # plot_histo(params['cond_counts'], params['group_names'], groups=params['cond_groups'],
    #            cond_names=params['cond_names'])

    plt.show()
