#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from design_optimisation.create_parameters_file import write_parameters_file

# First: Read the stimuli database infos and set some paradigm variables
stim_db = pd.read_csv("/hpc/banco/bastien.c/data/optim/calibrator_iti/"
                      "stim_db.csv", sep="\t")
files_list = stim_db['File']
durations = stim_db['Duration']
cond_of_files = stim_db['Condition']

iti_file = "/hpc/banco/bastien.c/data/optim/calibrator_iti/designs_iti_0_0/" \
           "ITIs.npy"
output_path = "/hpc/banco/bastien.c/data/optim/calibrator_iti/designs_iti_0_0"
nbr_designs = 100000

tr = 0.955


# Secondly, define tansitions table
conditions_names = [
    "old_F_high_nonspeech",
    "old_F_high_speech",
    "old_F_low_nonspeech",
    "old_F_low_speech",
    "old_M_high_nonspeech",
    "old_M_high_speech",
    "old_M_low_nonspeech",
    "old_M_low_speech",
    "young_F_high_nonspeech",
    "young_F_high_speech",
    "young_F_low_nonspeech",
    "young_F_low_speech",
    "young_M_high_nonspeech",
    "young_M_high_speech",
    "young_M_low_nonspeech",
    "young_M_low_speech",
]
groups = [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]
group_names = ["woman_nonspeech", "woman_speech", "man_nonspeech", "man_speech"]
tmp = np.array([
    [1],
    [2],
    [3],
    [4]
])

tmn = np.array([
    [0.1, 0.25, 0.25, 0.4],
    [0.25, 0.1, 0.4, 0.25],
    [0.25, 0.4, 0.1, 0.25],
    [0.4, 0.25, 0.25, 0.1]
])


contrasts = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 0],
    [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 0],
    [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0]
])
contrasts_names = [
    "Young vs. old",
    "Male vs. Female",
    "Low F0 vs. Hight F0",
    "Non Speech vs. Speech"
]

write_parameters_file(conditions_names, cond_of_files, groups, group_names,
                      contrasts, contrasts_names, durations, files_list,
                      iti_file, nbr_designs, tmp, tmn, tr, output_path,
                      question_dur=0)

