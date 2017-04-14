import numpy as np
import scipy.io as io

rowA = np.arange(1, 10)
rowB = np.arange(21, 25)
tab = np.empty((2,), dtype=object)
tab[0] = rowA
tab[1] = rowB
data = {"var": tab}


io.savemat("/hpc/banco/bastien.c/data/test_mat.mat", data)
