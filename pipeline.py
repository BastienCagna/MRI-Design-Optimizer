#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    MRI Design Optimisation Pipeline
    ================================

    This module create a large set of design and compute efficiencies for each design and contrasts

    Steps of this pipeline require that you previously created a parameters file using create_parameters_file.py

    :Example
    1 - Create the designs set:
    $ python design_effeciency.py generate_designs /hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/

    2 - Compute corresponding efficiencies (for all designs and all contrasts):
    $ python design_effeciency.py compute_efficiencies /hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/

    Todo:
    * Generalise the process to any type of stimulation files. This version can be used only with .wav file for audio
    stimulation.
"""

import sys
from os import system
import os.path as op
import pickle
import numpy as np

from design_efficiency import generate_designs, compute_efficiencies
from selection import find_best_designs, find_avg_designs
from viewer import plot_n_matrix, plot_distribs


def optimisation_process(params_path, n_sel):
    generate_designs(params_path, verbose=True)

    compute_efficiencies(params_path, verbose=True)

    find_best_designs(params_path, n=n_sel)

    find_avg_designs(params_path, n=n_sel, verbose=True)

    params = pickle.load(open(op.join(params_path, "params.p"), "rb"))
    path = params['output_path']

    best_indexes = np.load(op.join(path, "bests.npy"))
    plot_n_matrix(best_indexes, path, fig_file=op.join(path, "desgins_matrix_bests.png"))

    if not op.isdir(op.join(path, "distribs_bests")):
        system("mkdir {}".format(op.join(path, "distribs_bests")))
    for index in best_indexes:
        plot_distribs(op.join(path, "params.p"), op.join(path, "efficiencies.npy"), index,
                      op.join(path, "distribs_bests/{:05d}.png".format(index)))

    avgs_indexes = np.load(op.join(path, "avgs.npy"))
    plot_n_matrix(avgs_indexes, path, fig_file=op.join(path, "desgins_matrix_avgs.png"))

    if not op.isdir(op.join(path, "distribs_avgs")):
        system("mkdir {}".format(op.join(path, "distribs_avgs")))
    for index in avgs_indexes:
        plot_distribs(op.join(path, "params.p"), op.join(path, "efficiencies.npy"), index,
                      op.join(path, "distribs_avgs/{:05d}.png".format(index)))
    return


if __name__ == "__main__":
    p_path = sys.argv[1]
    n = int(sys.argv[2])

    optimisation_process(p_path, n)