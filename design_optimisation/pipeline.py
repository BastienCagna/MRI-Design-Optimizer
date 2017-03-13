#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    MRI Design Optimisation Pipeline
    ================================

"""

import os.path as op
import pickle
import sys
from os import system

import matplotlib
import numpy as np

matplotlib.use('Agg')

from design_optimisation.random_design_creator import generate_designs
from design_optimisation.design_efficiency import compute_efficiencies
from design_optimisation.selection import find_best_designs, find_avg_designs
from design_optimisation.viewer import plot_n_matrix, plot_distribs
from design_optimisation.export import to_labview


def optimisation_process(params_path, n_sel):
    """Design optimisation pipeline

    :param params_path: Path to the paramaters file
    :param n_sel: Number of design to select and export
    :return:
    """
    # Read parameters
    params = pickle.load(open(op.join(params_path, "params.pck"), "rb"))
    nbr_designs = params['nbr_designs']
    nbr_events = params['nbr_events']
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
    resp_dur = params['responses_dur']
    work_dir = params['work_dir']

    # Print some properties
    print("Parameters file path: {}".format(params_path))
    print("Working directory: {}".format(work_dir))
    print("ITI file: {}".format(iti_filename))
    print("\nNumber of designs: {}".format(nbr_designs))
    print("Number of conditions: {}".format(len(cond_counts)))
    print("Number of events: {}".format(nbr_events))

    # Create random designs (following transition probabilities)
    print("\n *** DESIGN CREATION ***")
    itis = np.load(iti_filename)
    designs, isi_maxs = generate_designs(nbr_designs, nbr_events, cond_counts, files_list, files_duration,
                                         cond_of_files, cond_names, cond_groups, group_names, tmp, tmn, itis,
                                         start_time=2.0, question_dur=resp_dur, verbose=True)
    pickle.dump({'designs': designs, 'isi_maxs': isi_maxs}, open(op.join(work_dir, "designs.pck"), "wb"))

    # Compute efficiencies of each design for each contrasts
    print("\n *** EFFICIENCIES COMPUTATION ***")
    efficiencies = compute_efficiencies(tr, designs, contrasts, isi_maxs, verbose=True)
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
    avgs_idx = find_avg_designs(efficiencies, contrasts, n=n_sel)
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

        to_labview(op.join(work_dir, "export/design_{:06d}.csv".format(idx)), designs[idx], question_cond="Question",
                   question_txt="+", question_dur=resp_dur, responses_idx=resp_v)

    return


if __name__ == "__main__":
    p_path = sys.argv[1]
    n = int(sys.argv[2])

    optimisation_process(p_path, n)
