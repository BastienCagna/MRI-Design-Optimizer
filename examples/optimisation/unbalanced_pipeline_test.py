import numpy as np

from design_optimisation.create_parameters_file import write_parameters_file

contrasts = [
    [-1, 1]
]

contrasts_names = ["Contrast 1"]

#  Transitions Constraints
groups = [1, 2]

tmp = [
    [1],
    [2]
]

C1 = 0.0
C2 = 0.0
tmn = [
    [C1, 1.0-C1],
    [1.0-C2, C2]
]

files_list = ["c1-1", "c1-2", "c1-3", "c1-4", "c1-5", "c1-6", "c1-7", "c1-8", "c1-9", "c1-10",
              "c2-1", "c2-2", "c2-3", "c2-4", "c2-5", "c2-6", "c2-7", "c2-8"]
cond_of_files = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
durations = np.repeat([0.5], 18)

conditions_names = ["cond1", "cond2"]

iti_file = "/hpc/banco/bastien.c/data/optim/test/ITIs.npy"
output_path = "/hpc/banco/bastien.c/data/optim/test/unbalanced/"
files_path = "./"

tr = 0.955
nbr_designs = 2000

write_parameters_file(conditions_names, cond_of_files, groups, contrasts, contrasts_names, durations, files_list,
                      files_path, iti_file, nbr_designs, tmp, tmn, tr, output_path, verbose=True)
