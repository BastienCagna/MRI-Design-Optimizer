from design_efficiency import write_parameters_file
import numpy as np
import scipy.io as io
import sys
import pandas as pd

# output_path = "/hpc/banco/bastien.c/data/optim/VL/banks/a100/params.p"
# tr = 0.975
# SOAmin = 4.5
# SOAmax = 8.0
#
# conditions_names = ['speech', 'emotional', 'non_speech', 'artificial', 'animal', 'environmental']
# count_by_cond = [24, 24, 24, 24, 24, 24]
# durations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#
# contrasts = np.array([[-1, -1, -1, 1, 1, 1, 0],
#                       [1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 1, 1, 1, 0],
#                       [-1, 1, 0, 0, 0, 0, 0],
#                       [-1, 0, 1, 0, 0, 0, 0],
#                       [0, -1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 1, -1, 0, 0],
#                       [0, 0, 0, 1, 0, -1, 0],
#                       [0, 0, 0, 0, 1, -1, 0]
#                     ])
#
# nbr_designs = 100000
#
# groups = [1, 1, 1, 2, 2, 2]
# alpha = 1.0
# tmp = np.array([
#     [1],
#     [2]
# ])
# tmn = np.array([
#     [1 - alpha, alpha],
#     [alpha, 1 - alpha]
# ])
# groups = [1, 2, 3, 4, 5, 6]
# tmp = np.array([
#     [1, 4],
#     [1, 5],
#     [1, 6],
#     [2, 4],
#     [2, 5],
#     [2, 6],
#     [3, 4],
#     [3, 5],
#     [3, 6],
#     [4, 1],
#     [4, 2],
#     [4, 3],
#     [5, 1],
#     [5, 2],
#     [5, 3],
#     [6, 1],
#     [6, 2],
#     [6, 3],
# ])
#
# tmn = np.array([
#     [0, 0.5, 0.5, 0, 0, 0],
#     [0, 0.5, 0.5, 0, 0, 0],
#     [0, 0.5, 0.5, 0, 0, 0],
#     [0.5, 0, 0.5, 0, 0, 0],
#     [0.5, 0, 0.5, 0, 0, 0],
#     [0.5, 0, 0.5, 0, 0, 0],
#     [0.5, 0.5, 0, 0, 0, 0],
#     [0.5, 0.5, 0, 0, 0, 0],
#     [0.5, 0.5, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.5, 0.5],
#     [0, 0, 0, 0, 0.5, 0.5],
#     [0, 0, 0, 0, 0.5, 0.5],
#     [0, 0, 0, 0.5, 0, 0.5],
#     [0, 0, 0, 0.5, 0, 0.5],
#     [0, 0, 0, 0.5, 0, 0.5],
#     [0, 0, 0, 0.5, 0.5, 0],
#     [0, 0, 0, 0.5, 0.5, 0],
#     [0, 0, 0, 0.5, 0.5, 0],
# ])


# IDENTIFCATION TASK -------
contrasts = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [-1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 0]
])
contrasts_names = ["Speaker 1 vs. Speaker 2",
                   "Speaker 2 vs. Speaker 3",
                   "Speaker 1 vs. Speaker 3",
                   "Low F0 vs. High F0"]

#  Transitions Constraints
groups = np.arange(1, 37)

Nspeak = 3
Nwords = 12

K = Nwords * Nspeak
p = 1 / ((Nspeak - 1)*(Nwords - 1))

tmp = []
for w1 in range(1, K+1):
    k = 0
    word1 = np.mod(w1, Nwords)
    for w2 in range(1, K+1):
        word2 = np.mod(w2, Nwords)
        if word2 != word1:
            tmp.append([w1, w2])
        else:
            k += 1
tmp = np.array(tmp)

speakers = np.repeat(np.arange(1, Nspeak+1), Nwords)
tmn = np.ones((K*(K-Nspeak), K))
for i, past in enumerate(tmp):
    w1 = past[0]
    w2 = past[1]
    spk1 = speakers[w1-1]
    spk2 = speakers[w2-1]
    for w3 in range(1, K+1):
        spk3 = speakers[w3-1]
        # No more than 2 time the same speaker
        if spk1 == spk3 and spk2 == spk3:
            tmn[i, w3-1] = 0
        # Never 2 time the same word
        elif np.mod(w3, Nwords) == np.mod(w2, Nwords):
            tmn[i, w3-1] = 0
sums = np.sum(tmn, axis=1)

for i in range(tmn.shape[0]):
    p = 1 / sums[i]
    for j in range(tmn.shape[1]):
        tmn[i, j] *= p

sound_list = io.loadmat("/hpc/banco/InterTVA/Sounds/Identification_task/List_identification.mat")
files_path = "/hpc/banco/InterTVA/Sounds/Identification_task/"
conditions_names = sound_list['List_identification'][:, 0]
durations = sound_list['List_identification'][:, 2] / 1000.0
files_list = []
for file in sound_list['List_identification'][:, 0]:
    files_list.append(file[0] + ".wav")
count_by_cond = np.ones((36,))
cond_of_files = np.arange(36)

iti_file = "/hpc/banco/bastien.c/data/optim/identification/ITIs.npy"
output_path = "/hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/"
tr = 0.955
nbr_designs = 100

write_parameters_file(conditions_names, cond_of_files, groups, contrasts, contrasts_names, files_list,
                      files_path, iti_file, nbr_designs, tmp, tmn, tr, output_path)
#  CALIBRATION TASK
# contrasts = np.array([
#     [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # Young / old
#     [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 0], # H / F
#     [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 0], # F0
#     [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0] # Speech / Non speech
# ])
# contrasts_names = ["Young vs. old", "Male vs. Female", "Low F0 vs. Hight F0", "Non Speech vs. Speech"]
#
# groups = [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]
#
# tmp = np.array([
#     [1],
#     [2],
#     [3],
#     [4]
# ])
#
# tmn = np.array([
#     [0.1, 0.25, 0.25, 0.4],
#     [0.25, 0.1, 0.4, 0.25],
#     [0.25, 0.4, 0.1, 0.25],
#     [0.4, 0.25, 0.25, 0.1]
# ])
#
# iti_file = "/hpc/banco/bastien.c/data/optim/VC/ITIs.npy"
# output_path = "/hpc/banco/bastien.c/data/optim/VC/iti_var/"
# nbr_designs = 5000
#
# output_params = output_path + "params.p"
# tr = 0.955
#
# cond_cond = np.load("/hpc/banco/bastien.c/data/optim/VC/cond_cond.npy")
# conditions_names = cond_cond[:, 0]
# durations = []
#
# write_parameters_file(output_params, conditions_names, cond_of_files, groups, contrasts, contrasts_names, files_list,
#                       files_path, iti_file, nbr_designs, tmp, tmn, output_path)
