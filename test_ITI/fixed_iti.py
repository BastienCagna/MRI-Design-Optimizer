import os.path as op
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from design_optimisation.design_efficiency import filtered_design_matrix, efficiency
from test_ITI.iti_test_common import set_fixed_iti, cut_design

# Plot options and config
sns.set(rc={'grid.color': 'darkgrey', 'grid.linestyle': ':', 'figure.figsize': [11, 7]})

N = 9
type = 'bests'

path = "/hpc/banco/bastien.c/data/optim/calibrator_iti/designs_iti_0_0/"
params_file = path + "params.pck"
designs_file = path + "designs.pck"
selection_file = path + "{}_idx.npy".format(type)
outpath = path + "out/"


# Read params file to get TR
params = pickle.load(open(params_file, "rb"))
tr = params['tr']
contrasts = params['contrasts']

initial_ITI = np.load(params['ITI_file'])
initial_ITI_avg = np.mean(initial_ITI)
initial_ITI_min = np.min(initial_ITI)
initial_ITI_max = np.max(initial_ITI)

# Read the selection
indexes = np.load(selection_file)
nbr_designs = indexes.shape[0]


# # Read designs and remove ITI
designs_sel = {}
isi_maxs = {}
print("Loading designs file...")
designs = pickle.load(open(designs_file, "rb"))
design_duration = 0
for i, idx in enumerate(indexes[:N]):
    # Remove iti and put the design in the selection dictionary
    designs_sel[idx] = set_fixed_iti(designs['designs'][idx], const_iti=0)

    # Compute max ISI (last ISI is not taken in account)
    onsets = designs['designs'][idx]['onset']
    isi_maxs[idx] = max(onsets[1:] - onsets[:-1])

    if design_duration < onsets[-1]:
        design_duration = onsets[-1]

    print("{} / {} design read".format(i + 1, N))
print("Design duration: {}".format(design_duration))

# Define tested ITI
nbr_ITI = 36
ITI_val = np.logspace(-2, 1.2, num=nbr_ITI)

for i_c in range(1, len(params['contrasts_names'])):
    c = contrasts[i_c]
    contrast_name = params['contrasts_names'][i_c]
    # c = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    # contrast_name = "all_minus_silence"
    print(contrast_name)

    # Compute efficiency obtained for each ITI on each selected designs with 3 different filtering (without, bandpass,
    # highpass)
    efficiencies = []
    hp_cut_freq = []
    for i, iti in enumerate(ITI_val):
        print("ITI {: 2d}/{} : {}".format(i+1, nbr_ITI, iti))
        for idx in indexes[:N]:
            # Create a design for each iti value
            # tmp_design = set_fixed_iti(ref_design, iti)
            tmp_design = set_fixed_iti(designs_sel[idx], iti)

            tmp_design = cut_design(tmp_design, design_duration)
            isi_max = isi_maxs[idx] + iti
            hp_cut_freq.append(1 / isi_max)

            X = filtered_design_matrix(tr, tmp_design, isi_max, filt_type='highpass')

            # Compute efficiency of this design for this contrast
            eff = efficiency(X, c)
            efficiencies.append([iti, idx, eff])
    efficiencies = np.array(efficiencies)

    # Save efficiencies
    np.save(op.join(outpath, "data_{}_{}_{}".format(type, N, contrast_name)), efficiencies)
    # efficiencies = np.load(op.join(outpath, "data_{}_{}.npy".format(type, N)))

    # Search the best efficiency of each design
    max_eff = np.zeros((N,))
    for i, idx in enumerate(indexes[:N]):
        max_eff[i] = max(efficiencies[efficiencies[:, 1] == idx, 2])
    # Order N first indexes in a way that design's efficiencies will be plotted from the worst
    # design to the best
    ordered_indexes = indexes[np.argsort(max_eff)]

    # Plot
    fig, ax = plt.subplots()

    cmap = sns.color_palette("husl", N) # sns.cubehelix_palette(N)
    for i, idx in enumerate(ordered_indexes):
        plt.plot(efficiencies[efficiencies[:, 1] == idx, 0], efficiencies[efficiencies[:, 1] == idx, 2], color=cmap[i],
                 zorder=i+3)

    lcolor = 'silver'
    plt.plot([initial_ITI_min, initial_ITI_min], plt.ylim(), color=lcolor, linestyle='--', zorder=0)
    plt.plot([initial_ITI_avg, initial_ITI_avg], plt.ylim(), color=lcolor, linestyle='-', zorder=1)
    plt.plot([initial_ITI_max, initial_ITI_max], plt.ylim(), color=lcolor, linestyle='--', zorder=2)

    plt.xlabel("ITI (s)")
    plt.xscale('log')
    plt.ylabel("Efficiency (arbitrary)")
    plt.legend(ordered_indexes)
    plt.title("Efficiency vs. fixed ITI duration")

    fig.savefig(op.join(outpath, "eff_vs_iti_{}_{}_{}.png".format(type, N, contrast_name)))
# print("Figure saved to: {}".op.join(outpath, "eff_vs_iti_{}_{}_designs.png".format(type, N)))
#plt.show()
