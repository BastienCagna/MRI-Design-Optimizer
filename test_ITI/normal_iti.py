import os.path as op
import pickle

import matplotlib.pyplot as plt
import numpy as np

from design_optimisation.design_efficiency import filtered_design_matrix, efficiency
from test_ITI.iti_test_common import set_fixed_iti, set_random_iti

inpath = "/hpc/banco/bastien.c/data/optim/calibrator_iti"
outpath = "/hpc/banco/bastien.c/data/optim/calibrator_iti/normal_distrib"

# Read params file to get TR
params = pickle.load(open(op.join(inpath, "params.pck"), "rb"))
tr = params['tr']
contrasts = params['contrasts']
c = contrasts[0]

# Read the reference design
design = pickle.load(open(op.join(inpath, "ref_design.pck"), "rb"))
print("Design read from: {}".format(op.join(inpath, "ref_design.pck")))

# Remove itis
ref_design = set_fixed_iti(design, const_iti=0)

# Save this design
pickle.dump( ref_design, open(op.join(outpath, "no_iti_design.pck"), "wb"))
print("Design without iti saved in: {}".format(op.join(outpath, "no_iti_design.pck")))

# Search the maxima duration
dur_max = max(ref_design['duration'])
print("Maximal duration is: {}".format(dur_max))

# Compute efficiency for different iti value
ITI_avg = [0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
ITI_std = [0, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
efficiencies = []
hp_cut_freq = []
for iti_avg in ITI_avg:
    for iti_std in ITI_std:
        # Create a design for each iti value
        # tmp_design = set_fixed_iti(ref_design, iti)
        tmp_design = set_random_iti(ref_design, iti_avg, 3.0)

        isi_max = dur_max + iti_avg
        hp_cut_freq.append(1 / isi_max)

        Xpass = filtered_design_matrix(tr, tmp_design, isi_max, filt_type='pass')
        # Xband = filtered_design_matrix(tr, tmp_design, isi_max, nf=6)
        # Xhpass = filtered_design_matrix(tr, tmp_design, isi_max, filt_type='highpass')

        # Compute efficiency of this design for this contrast
        e_pass = efficiency(Xpass, c)
        # e_band = efficiency(Xband, c)
        # e_hpass = efficiency(Xhpass, c)
        efficiencies.append([iti_avg, iti_std, e_pass])#, e_band, e_hpass])
        print("Efficiency with iti of {:.3f}s +/- {:.3f}= {:.4f}".format(iti_avg, iti_std, e_pass))
efficiencies = np.array(efficiencies)
# efficiencies = np.load(op.join(outpath, "efficiencies.npy"))

# Save efficiencies
np.save(op.join(outpath, "efficiencies.npy"), efficiencies)
print("Efficiencies saved to: {}".format(op.join(outpath, "efficiencies.npy")))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(efficiencies[:, 0], efficiencies[:, 1], efficiencies[:, 2], c="b")
# ax.scatter(efficiencies[:, 0], efficiencies[:, 1], efficiencies[:, 3], c='r')
# ax.scatter(efficiencies[:, 0], efficiencies[:, 1], efficiencies[:, 4], c="g")
ax.set_xlabel("ITI avg (s)")
ax.set_ylabel("ITI std (s)")
plt.legend(['No filter', 'Pass band', 'High pass'])
ax.set_zlabel("Efficiency (arbitrary)")
ax.set_title("Efficiency vs. fixed ITI duration")

fig.savefig(op.join(outpath, "eff_vs_iti.png"))
print("Figure saved to: {}".format(op.join(outpath, "eff_vs_iti.png")))
plt.show()
