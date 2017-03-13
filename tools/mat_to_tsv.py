#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from scipy import io
import numpy as np
import sys


def mat_to_csv(mat_filename, csv_filename, mat_varname, columns_names):
    mat_file = io.loadmat(mat_filename)

    data = mat_file[mat_varname]
    nbr_row = data.shape[0]
    nbr_col = data.shape[1]

    flatten_data = []
    for i in range(nbr_row):
        row = []
        for j in range(nbr_col):
            tmp = data[i][j]
            # Avoid any array encapsulation
            while isinstance(tmp, (list, tuple, np.ndarray)):
                tmp = tmp[0]
            row.append(tmp)
        flatten_data.append(row)

    df = pd.DataFrame(flatten_data, columns=columns_names)
    df.to_csv(csv_filename, index=False, sep="\t")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("\nConvert .mat file to .tsv\n")
        print("--- Arguments ---")
        print("  [1]  Path to the .mat file\n  [2]  Path the new .tsv file")
        print("  [3]  Name of the variable contained by the .mat file")
        print("  [4+] Columns names of the tsv file separated by whitespaces")
        print("\n--- Return ---\n  Create the new file corresponding to the given output filename.\n")
        exit()

    cols_names = []
    for i in range(4, len(sys.argv)):
        cols_names.append(sys.argv[i])

    mat_to_csv(sys.argv[1], sys.argv[2], sys.argv[3], cols_names)

