import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os.path as op

from design_optimisation.design_efficiency import filtered_design_matrix, efficiency
from test_ITI.iti_test_common import set_fixed_iti

# Plot options and config
sns.set(rc={'grid.color': 'darkgrey', 'grid.linestyle': ':', 'figure.figsize': [11, 7]})


path = "/hpc/banco/bastien.c/data/optim/calibrator_iti/designs_iti_0_0/"
params_file = path + "params.pck"
designs_file = path + "designs.pck"
selection_file = path + "{}_idx.npy".format(type)
outpath = path + "out/"

N = 9
contrasts = ["Young vs. old", "Male vs. Female", "Low F0 vs. Hight F0", "Non Speech vs. Speech" ]
best_c = 'silver'
avg_c = 'lightgray'
sel_c ='goldenrod'
idx_sel = 92632
plt.figure()
for j, contrast_name in enumerate(contrasts):
    best_eff = np.load(op.join(outpath, "data_bests_{}_{}.npy".format(N, contrast_name)))
    indexes_best = np.array(np.unique(best_eff[:, 1]), dtype=int)
    avg_eff = np.load(op.join(outpath, "data_avgs_{}_{}.npy".format(N, contrast_name)))
    indexes_avg = np.array(np.unique(avg_eff[:, 1]), dtype=int)

    plt.subplot(2, 2, j+1)
    for i, idx in enumerate(indexes_avg):
        plt.plot(avg_eff[avg_eff[:, 1] == idx, 0], avg_eff[avg_eff[:, 1] == idx, 2],
                 color=avg_c)

        idx = indexes_best[i]
        plt.plot(best_eff[best_eff[:, 1] == idx, 0], best_eff[best_eff[:, 1] == idx, 2],
                 color=best_c)

        plt.plot(best_eff[best_eff[:, 1] == idx_sel, 0], best_eff[best_eff[:, 1] == idx_sel, 2],
                 color=sel_c)

        plt.xscale("log")
        plt.title(contrast_name)
plt.show()
