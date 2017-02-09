from create_parameters_file import write_parameters_file
import numpy as np


contrasts = [
    [-1, -1, -1, 1, 1, 1, 0],
    [-1, 1, -1, 1, -1, 1, 0],
    [-1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0],
]

contrasts_names = ["Contrast 1", "Contrast 2", "Contrast 3", "Contrast 4"]

#  Transitions Constraints
groups = [1, 1, 1, 2, 2, 2]

tmp = [
    [1],
    [2]
]

tmn = [
    [0.0, 1.0],
    [1.0, 0.0]
]

files_list = ["c1-1", "c1-2", "c1-3", "c2-1", "c2-2", "c2-3", "c3-1", "c3-2", "c3-3", "c4-1", "c4-2", "c4-3",
              "c5-1", "c5-2", "c5-3", "c6-1", "c6-2", "c6-3"]
cond_of_files = np.repeat([0, 1, 2, 3, 4, 5], 3)
durations = np.repeat([0.5], 18)

conditions_names = ["cond1", "cond2", "cond3", "cond4", "cond5", "cond6"]

iti_file = "/hpc/banco/bastien.c/data/optim/test/ITIs.npy"
output_path = "/hpc/banco/bastien.c/data/optim/test/balanced/"
files_path = "./"

tr = 0.955
nbr_designs = 2000

write_parameters_file(conditions_names, cond_of_files, groups, contrasts, contrasts_names, durations, files_list,
                      files_path, iti_file, nbr_designs, tmp, tmn, tr, output_path, verbose=True)
