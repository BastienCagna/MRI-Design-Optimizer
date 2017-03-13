import os.path as op
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from design_optimisation.design_efficiency import filtered_design_matrix, efficiency
from test_ITI.iti_test_common import set_fixed_iti

params_file = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/calibrator/designs/params.pck"
designs_file = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/calibrator/designs/designs.pck"
selection_file = "/hpc/banco/bastien.c/data/optim/calibrator_iti/large_fixed_iti/designs_indexes.npy"
outpath = "/hpc/banco/bastien.c/data/optim/calibrator_iti/large_fixed_iti"


# Read params file to get TR
params = pickle.load(open(params_file, "rb"))
tr = params['tr']
contrasts = params['contrasts']
c = contrasts[0]


# Read the selection
indexes = np.load(selection_file)
nbr_designs = indexes.shape[0]


# Read designs and remove ITI
designs_sel = {}
isi_maxs = {}
designs = pickle.load(open(designs_file, "rb"))
for idx in indexes:
    # Remove iti and put the design in the selection dictionary
    designs_sel[idx] = set_fixed_iti(designs['designs'][idx], const_iti=0)
    # Compute max ISI (last ISI is not taken in account)
    onsets = designs['designs'][idx]['onset']
    isi_maxs[idx] = max(onsets[1:] - onsets[:-1])


# Define tested ITI
nbr_ITI = 15
ITI_val = np.logspace(-2, 1.2, num=nbr_ITI)


# Compute efficiency obtained for each ITI on each selected designs with 3 different filtering (without, bandpass,
# highpass)
efficiencies = []
hp_cut_freq = []
for iti in ITI_val:
    print("ITI: {}".format(iti))
    for idx in indexes:
        # Create a design for each iti value
        # tmp_design = set_fixed_iti(ref_design, iti)
        tmp_design = set_fixed_iti(designs_sel[idx], iti)

        isi_max = isi_maxs[idx] + iti
        hp_cut_freq.append(1 / isi_max)

        Xpass = filtered_design_matrix(tr, tmp_design, isi_max, filt_type='pass')
        Xband = filtered_design_matrix(tr, tmp_design, isi_max, nf=6)
        Xhpass = filtered_design_matrix(tr, tmp_design, isi_max, filt_type='highpass')

        # Compute efficiency of this design for this contrast
        efficiencies.append([iti, idx, "no filtering", efficiency(Xpass, c)])
        efficiencies.append([iti, idx, "band pass", efficiency(Xband, c)])
        efficiencies.append([iti, idx, "high pass", efficiency(Xhpass, c)])
efficiencies = np.array(efficiencies)
data = {"ITI": efficiencies[:, 0], "Design": efficiencies[:, 1], "Filtering": efficiencies[:, 2],
        "Efficiency": efficiencies[:, 3]}

# Save efficiencies
# pickle.dump(data, open(op.join(outpath, "data.pck"), "wb"))
data = pickle.load(open(op.join(outpath, "data.pck"), "rb"))

iti_v = np.array(data['ITI'], dtype=float)
des_v = np.array(data['Design'], dtype=int)
eff_v = np.array(data['Efficiency'], dtype=float)
filt_v = np.array(data['Filtering'])
data = {"ITI": iti_v, "Design": des_v, "Filtering": data['Filtering'], "Efficiency": eff_v}

stats = []
for iti in np.unique(iti_v):
    for filt in np.unique(filt_v):
        avg = np.mean(eff_v[(iti_v == iti) * (filt_v == filt)])
        std = np.mean(eff_v[(iti_v == iti) * (filt_v == filt)])
        stats.append([iti, filt, avg, std])
stats = np.array(stats)


# Plot
fig, ax = plt.subplots()
cmap = sns.color_palette("hls", 3)
# plt.plot(stats[stats[:, 1] == "no filtering", 0], stats[stats[:, 1] == "no filtering", 2], color=cmap[0])
# plt.plot(stats[stats[:, 1] == "Band pass", 0], stats[stats[:, 1] == "Band pass", 2], color=cmap[0])
# plt.plot(stats[stats[:, 1] == "High pass", 0], stats[stats[:, 1] == "High pass", 2], color=cmap[0])
sns.boxplot(x='ITI', y="Efficiency", hue="Filtering", data=data, palette="Set2")
# plt.xticks(rotation='vertical')
# plt.subplots_adjust(bottom=0.25)
plt.xlabel("ITI (s)")
# plt.xscale('log')
# plt.legend(['No filter', 'Pass band', 'High pass'])
plt.ylabel("Efficiency (arbitrary)")
plt.title("Efficiency vs. fixed ITI duration")

fig.savefig(op.join(outpath, "eff_vs_iti.png"))
print("Figure saved to: {}".format(op.join(outpath, "eff_vs_iti.png")))
plt.show()
