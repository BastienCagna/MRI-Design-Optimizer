#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from design_optimisation.create_parameters_file import write_parameters_file

# IDENTIFICATION TASK
# The identification task is composed of 12 different words pronounced by 3 speakers. Each speaker pronounces the
# same 12 words. Thus, there is 36 waves files (3x12). Each file duration is different.
#
# The design allows to have two consecutive time the same speaker but never two times the same word.
#

# First: Read the stimuli database infos and set some paradigm variables

stim_db = pd.read_csv("/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/identification_task/stim_db.csv",
                      sep="\t")
files_list = stim_db['File']
durations = stim_db['Duration']
conditions_names = stim_db['Condition']

count_by_cond = np.ones((36,))
cond_of_files = np.arange(36)
iti_file = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/identification_task/itis.npy"
output_path = "/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/identification_task/new/"
tr = 0.955
nbr_designs = 100000


# Secondly, define tansitions table
# Number of speakers
Nspeak = 3
# Number of different words (each speaker pronounces the same words)
Nwords = 12
# Total number of words (or conditions)
K = Nwords * Nspeak


# Conditions are not grouped, so groups is a vector starting from 1 to 36
groups = np.arange(1, K+1)
group_names = np.repeat(["Anne", "Betty", "Chloe"], Nwords)
responses_of_cond = np.repeat([-1, -2, -3], Nwords)


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


write_parameters_file(conditions_names, cond_of_files, groups, group_names, contrasts, contrasts_names, durations,
                      files_list, iti_file, nbr_designs, tmp, tmn, tr, output_path, responses=responses_of_cond,
                      question_dur=5.0)

