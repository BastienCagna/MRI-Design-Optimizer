import os.path as op
import pickle

import matplotlib.pyplot as plt
import numpy as np
from nistats.design_matrix import plot_design_matrix

from design_optimisation.design_efficiency import design_matrix

#
# labview = pd.read_csv("/hpc/banco/InterTVA/bastien/sub-01/Sub-01_func07_SequenceVoice_calibrator_V0_17_01_25_15_41"
#                       ".txt", sep="\t")
#
# db = pd.read_csv("/hpc/banco/InterTVA/bastien/calibrator_stim_db.tsv", sep="\t")
#
# for i, stim in enumerate(labview['CONDITION']):
#     if stim == "Stim":
#         idx_db = np.argwhere(db['File'] == labview['SON'][i]).flatten()[0]
#         labview['CONDITION'][i] = db['Condition'][idx_db]
#
# labview.to_csv("/hpc/banco/InterTVA/bastien/sub-01/Sub-01_func07_SequenceVoice_calibrator_V0_17_01_25_15_41_nouveau"
#                ".txt", sep="\t", index=False)

path = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/identification_task/past/"
# path = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/calibrator/designs/"
efficiencies = np.load(op.join(path, "efficiencies.npy")).T
data = pickle.load(open(op.join(path, "designs.pck"), "rb"))
designs = data['designs']

print("nbr design: {}\nnbr efficiencies: {}".format(len(designs), len(efficiencies)))
b_indexes = np.load(op.join(path, "bests_idx.npy"))
a_indexes = np.load(op.join(path, "avgs_idx.npy"))
design = designs[58393]

# # Real design
# x = design_matrix(design, 0.955)
# plot_design_matrix(x)
# plt.title("Original design matrix")

# With grouped conditions
t_type = []
for cond in design['trial_type']:
    t_type.append(cond[:5])
design['trial_type'] = t_type
x = design_matrix(design, 0.955)
plot_design_matrix(x)
plt.show("Grouped condition design matrix")


# indexes = [a_indexes[0], b_indexes[0], a_indexes[1], b_indexes[1], a_indexes[2], b_indexes[2]]
#
# plt.figure()
# for i, idx in enumerate(indexes):
#     ax = plt.subplot(3, 2, i+1)
#     design = designs[idx]
#     t_type = []
#     for cond in design['trial_type']:
#         t_type.append(cond[:5])
#     design['trial_type'] = t_type
#     x = design_matrix(design, 0.955)
#     plot_design_matrix(x, ax=ax)
    #
    # plt.text(2.5, 150, "e1 = {:.04f}".format(efficiencies[idx, 0]), color="m", fontsize=12)
    # plt.text(2.5, 180, "e2 = {:.04f}".format(efficiencies[idx, 1]), color="m", fontsize=12)
    # plt.text(2.5, 210, "e3 = {:.04f}".format(efficiencies[idx, 2]), color="m", fontsize=12)
    # plt.text(2.5, 240, "e4 = {:.04f}".format(efficiencies[idx, 3]), color="m", fontsize=12)

# params = pickle.load(open("/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/calibrator"
#                           "/designs/params.pck", 'rb'))
#
# print(params['cond_counts'])
# plt.figure()
# plt.bar(np.arange(16), params['cond_counts'], np.ones((16,))*0.8)
# plt.title("Histogram of event's condition")

plt.tight_layout()
plt.show()
