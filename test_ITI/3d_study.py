import pickle
import sys
import time

import numpy as np
import seaborn as sns

sys.path.append("/hpc/banco/bastien.c/python/design_optimizer")

from test_ITI.iti_test_common import set_fixed_iti
from design_optimisation.design_efficiency import filtered_design_matrix, efficiency

# Plot options and config
sns.set(rc={'grid.color': 'darkgrey', 'grid.linestyle': ':', 'figure.figsize': [11, 7]})


params_file = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/calibrator/designs/params.pck"
designs_file = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/calibrator/designs/designs.pck"
output_file = "/hpc/banco/bastien.c/data/optim/calibrator_iti/3d_study/efficiencies.npy"


# Read params file to get TR
params = pickle.load(open(params_file, "rb"))
tr = params['tr']
contrasts = params['contrasts']

initial_ITI = np.load(params['ITI_file'])
initial_ITI_avg = np.mean(initial_ITI)
initial_ITI_min = np.min(initial_ITI)
initial_ITI_max = np.max(initial_ITI)


# Read designs and remove ITI
designs_no_ITI = []
isi_maxs = {}
print("{}: Loading designs file...".format(time.ctime()))
designs = pickle.load(open(designs_file, "rb"))
nbr_designs = len(designs['designs'])
print("\t{} designs read".format(nbr_designs))

print("{}: Removing initial ITIs...".format(time.ctime()))
for i, design in enumerate(designs['designs']):
    if np.mod(i, nbr_designs/20) == 0:
        print("\t{:.01f}%".format(100 * (i/nbr_designs)))

    # Remove iti and put the design in the selection dictionary
    designs_no_ITI.append(set_fixed_iti(design, const_iti=0))
    # Compute max ISI (last ISI is not taken in account)
    onsets = designs_no_ITI[i]['onset']
    isi_maxs[i] = max(onsets[1:] - onsets[:-1])


# Define tested ITI
nbr_ITI = 50
ITI_val = np.logspace(-2, 1.2, num=nbr_ITI)


# Compute efficiency obtained for each ITI on each selected designs with 3 different filtering (without, bandpass,
# highpass)
efficiencies = []
hp_cut_freq = []
eff = np.zeros((len(contrasts), ))

print("{}: Computing efficiencies...".format(time.ctime()))
for i, iti in enumerate(ITI_val):
    print("{}: \tITI {: 2d}/{} : {}".format(time.ctime(), i+1, nbr_ITI, iti))

    for idx, design in enumerate(designs_no_ITI):
        if np.mod(idx, nbr_designs/200) == 0:
            print("{}\t\t{:.01f}%".format(time.ctime(), 100 * (idx/nbr_designs)))

        # Create a design for each iti value
        tmp_design = set_fixed_iti(design, iti)

        isi_max = isi_maxs[idx] + iti
        hp_cut_freq.append(1 / isi_max)

        X = filtered_design_matrix(tr, tmp_design, isi_max, filt_type='highpass')

        # Compute efficiency of this design for this contrast
        for ic, c in enumerate(contrasts):
            eff[ic] = efficiency(X, c)
        efficiencies.append([idx, iti, np.mean(eff), np.std(eff)])
efficiencies = np.array(efficiencies)

print("{}: Data have been saved to {}".format(time.ctime(), output_file))
np.save(output_file, efficiencies)