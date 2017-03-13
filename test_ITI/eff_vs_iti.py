import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from design_optimisation.design_efficiency import filtered_design_matrix, efficiency
from test_ITI.iti_test_common import set_fixed_iti

# Plot options and config
sns.set(rc={'grid.color': 'darkgrey', 'grid.linestyle': ':', 'figure.figsize': [11, 7]})


path = "/hpc/banco/bastien.c/data/optim/calibrator_iti/designs_iti_0_0_tmp/"
params_file = path + "params.pck"
designs_file = path + "designs.pck"
selection_file = path + "{}_idx.npy".format(type)
outpath = path + "out/"


# Read params file to get TR
params = pickle.load(open(params_file, "rb"))
tr = params['tr']
contrasts = params['contrasts']
c = contrasts[0]

design_idx = 1

# Read designs
print("Loading designs file...")
data = pickle.load(open(designs_file, "rb"))
isi_maxs = data["isi_maxs"]
isi_max = isi_maxs[design_idx]
designs = data['designs']
design = designs[design_idx]


# Define tested ITI
nbr_ITI = 36
ITI_val = np.logspace(-2, 1.2, num=nbr_ITI)


# Compute efficiency for each ITI
efficiencies = []
hp_cut_freq = []
for i, iti in enumerate(ITI_val):
    print("ITI {: 2d}/{} : {}".format(i+1, nbr_ITI, iti))
    # Create a design for each iti value
    # tmp_design = set_fixed_iti(ref_design, iti)
    tmp_design = set_fixed_iti(design, iti)

    isi_m = isi_max + iti
    hp_cut_freq.append(1 / isi_m)

    X = filtered_design_matrix(tr, tmp_design, isi_m, filt_type='highpass')

    # Compute efficiency of this design for this contrast
    eff = efficiency(X, c)
    efficiencies.append([iti, eff])
efficiencies = np.array(efficiencies)


# Plot
fig, ax = plt.subplots()
plt.plot(efficiencies[:, 0], efficiencies[:, 1])
plt.xlabel("ITI (s)")
plt.xscale('log')
plt.ylabel("Efficiency (arbitrary)")
plt.title("Efficiency vs. fixed ITI duration")

plt.show()
