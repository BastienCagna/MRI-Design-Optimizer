#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


output_path = "/hpc/banco/bastien.c/data/optim/VL/banks/a100/params.p"
tr = 0.975

conditions_names = ['speech', 'emotional', 'non_speech', 'artificial', 'animal', 'environmental']
count_by_cond = [24, 24, 24, 24, 24, 24]
durations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

contrasts = np.array([[-1, -1, -1, 1, 1, 1, 0],
                      [1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0],
                      [-1, 1, 0, 0, 0, 0, 0],
                      [-1, 0, 1, 0, 0, 0, 0],
                      [0, -1, 1, 0, 0, 0, 0],
                      [0, 0, 0, -1, 1, 0, 0],
                      [0, 0, 0, -1, 0, 1, 0],
                      [0, 0, 0, 0, -1, 1, 0]
                    ])
contrasts_names = [
    "Voice vs. Non-Voice",
    "Voice effect",
    "Non-Voice effect",
    "Speech vs. Emotional",
    "Speech vs. Non-speech",
    "Artificial vs. Animal",
    "Artificial vs. Environmental",
    "Animal vs. Environmental"
]

nbr_designs = 100000

groups = [1, 1, 1, 2, 2, 2]
alpha = 1.0
tmp = np.array([
    [1],
    [2]
])
tmn = np.array([
    [1 - alpha, alpha],
    [alpha, 1 - alpha]
])
groups = [1, 2, 3, 4, 5, 6]
tmp = np.array([
    [1, 4],
    [1, 5],
    [1, 6],
    [2, 4],
    [2, 5],
    [2, 6],
    [3, 4],
    [3, 5],
    [3, 6],
    [4, 1],
    [4, 2],
    [4, 3],
    [5, 1],
    [5, 2],
    [5, 3],
    [6, 1],
    [6, 2],
    [6, 3],
])

tmn = np.array([
    [0, 0.5, 0.5, 0, 0, 0],
    [0, 0.5, 0.5, 0, 0, 0],
    [0, 0.5, 0.5, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0],
    [0.5, 0.5, 0, 0, 0, 0],
    [0.5, 0.5, 0, 0, 0, 0],
    [0.5, 0.5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.5, 0.5],
    [0, 0, 0, 0, 0.5, 0.5],
    [0, 0, 0, 0, 0.5, 0.5],
    [0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0.5, 0.5, 0],
    [0, 0, 0, 0.5, 0.5, 0],
    [0, 0, 0, 0.5, 0.5, 0],
])


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
