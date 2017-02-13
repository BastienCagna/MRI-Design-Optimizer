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
from export import to_labview


def optimisation_process(params_path, n_sel):
    # Read parameters
    params = pickle.load(open(op.join(params_path, "params.pck"), "rb"))
    nbr_designs = params['nbr_designs']
    cond_counts = params['cond_counts']
    files_list = params['files_list']
    files_duration = params['files_duration']
    cond_of_files = params['cond_of_files']
    cond_groups = params['cond_groups']
    cond_names = params['cond_names']
    group_names = params['group_names']
    iti_filename = params['ITI_file']
    tmp = params['TMp']
    tmn = params['TMn']
    tr = params['tr']
    contrasts = params['contrasts']
    contrasts_names = params['contrasts_names']
    responses = params['responses']
    response_dur = params['responses_dur']
    work_dir = params['work_dir']

    # Create random designs (following transition probabilities)
    print("\n *** DESIGN CREATION ***")
    designs, isi_maxs = generate_designs(nbr_designs, cond_counts, files_list, files_duration, cond_of_files,
                                         cond_names, cond_groups, group_names, tmp, tmn, iti_filename, start_time=2.0,
                                         verbose=True)
    pickle.dump({'desgins': designs, 'isi_maxs': isi_maxs}, open(op.join(work_dir, "designs.pck"), "wb"))

    # Compute efficiencies of each design for each contrasts
    print("\n *** EFFICIENCIES COMPUTATION ***")
    efficiencies = compute_efficiencies(tr, designs, contrasts, isi_maxs, cond_names, verbose=True)
    np.save(op.join(work_dir, "efficiencies.npy"), efficiencies)

    # Choose and plot best designs
    print("\n *** SEARCHING FOR BEST DESGIN(S) ***")
    bests_idx = find_best_designs(efficiencies, contrasts, n=n_sel)
    np.save(op.join(work_dir, "bests_idx.npy"), bests_idx)

    plot_n_matrix(bests_idx, designs, tr, fig_file=op.join(work_dir, "desgins_matrix_bests.png"))

    system("mkdir {}".format(op.join(work_dir, "distribs_bests")))
    for idx in bests_idx:
        plot_distribs(idx, efficiencies, contrasts_names, fig_file=op.join(work_dir, "distribs_bests/{:06d}".format(
            idx)))

    # Choose and plot "Average" designs
    print("\n *** SEARCHING FOR AVG DESGIN(S) ***")
    avgs_idx = find_avg_designs(efficiencies, contrasts, n=n_sel, verbose=True)
    np.save(op.join(work_dir, "avgs_idx.npy"), avgs_idx)

    plot_n_matrix(avgs_idx, designs, tr, fig_file=op.join(work_dir, "desgins_matrix_avgs.png"))

    system("mkdir {}".format(op.join(work_dir, "distribs_avgs")))
    for idx in avgs_idx:
        plot_distribs(idx, efficiencies, contrasts_names, fig_file=op.join(work_dir, "distribs_avgs/{:06d}".format(
            idx)))

    # Exportation to labview
    print("\n *** CSV EXPORTATION ***")
    print("Best design exportation...")
    system("mkdir {}".format(op.join(work_dir, "export")))
    for idx in bests_idx:
        if responses is not None:
            resp_v = [responses[j] for j in designs['trial_idx']]
        else:
            resp_v = None
            response_dur = 0

        to_labview(op.join(work_dir, "export/design_{:06d}.csv"), designs[idx], question_cond="Question",
                   question_txt="+", question_dur=response_dur, responses_idx=resp_v)

    return


if __name__ == "__main__":
    p_path = sys.argv[1]
    n = int(sys.argv[2])

    optimisation_process(p_path, n)
