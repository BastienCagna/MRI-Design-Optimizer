#!/usr/bin/env python
# -*- coding: utf-8 -*-

from create_parameters_file import write_parameters_file
import numpy as np
import scipy.io as io
import os.path as op
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

# IDENTIFICATION TASK
# The identification task is composed of 12 different words pronounced by 3 speakers. Each speaker pronounces the
# same 12 words. Thus, there is 36 waves files (3x12). Each file duration is different.
#
# The design allows to have two consecutive time the same speaker but never two times the same word.
#

# First: Read the stimuli database infos and set some paradigm variables
stim_db = pd.read_csv("/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/identification_task/stim_db.csv")
files_list = stim_db['File']
durations = stim_db['Duration']
conditions_names = ['Condition']

count_by_cond = np.ones((36,))
cond_of_files = np.arange(36)
iti_file = "/hpc/banco/bastien.c/data/optim/identification/ITIs.npy"
output_path = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/identification_task/"
tr = 0.955
nbr_designs = 500


# Secondary, define tansitions table

# Conditions are not grouped, so groups is a vector starting from 1 to 36
groups = np.arange(1, 37)

# Number of speakers
Nspeak = 3
# Number of different words (each speaker pronounces the same words)
Nwords = 12
# Total number of words (or conditions)
K = Nwords * Nspeak

# TMp matrix contains all possible couple of condition excepted two time the same word (regardless of the speaker)
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

# For each allowed couple of words, set probabilities to access any words according to the constraint (never two
# times the same word and never more than two times the same speaker)
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

# For each rows, allowed transition are currently set to 1. Now, set value in probalities. Each allowed transitions
# are equiprobable.
for i in range(tmn.shape[0]):
    p = 1 / sums[i]
    for j in range(tmn.shape[1]):
        tmn[i, j] *= p


# Finally, define the interresting contrasts
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


write_parameters_file(conditions_names, cond_of_files, groups, contrasts, contrasts_names, durations, files_list,
                      iti_file, nbr_designs, tmp, tmn, tr, output_path, verbose=True)
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
