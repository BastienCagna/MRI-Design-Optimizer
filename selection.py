#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Desgins Selection
    =================

    This module create a large set of design and compute efficiencies for each design and contrasts

    Functions of this file require that you previously created a parameters file using create_parameters_file.py and
    create a set of designs and compute their efficiencies for each contrasts using design_efficiency.py.

    :Example
    * Find the best designs (only one or several, here the 9 best):
    $ python selection.py find_best_desings /hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/ 9

    * Find some 'average' designs (only one or several, here the 9 best):
    $ python selection.py find_avg_desings /hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/ 9

"""

import sys
import numpy as np
import pickle
import warnings
import os.path as op


def find_best_designs(params_path, params_file="params.p", efficiencies_file="efficiencies.npy",
                      output_file="bests.npy", n=1):
    """Return one or several index(es) of designs that provide jointly best efficiencies for all contrasts.

    :param params_path: Path of the directory that contains the parameters file (by default, this is the output dir.).
    :param params_file: (optional) Parameters file name: Default: params.p
    :param efficiencies_file: (optional) Efficiencies file name. Default: efficiencies.npy
    :param n: (optional) Number of indexes to return. Default: 1.
    :type n: int
    :return: Nothing.
    """
    # Load data
    params = pickle.load(open(op.join(params_path, params_file), "rb"))
    contrasts = params['contrasts']
    efficiencies = np.load(op.join(params['output_path'], efficiencies_file))

    nbr_tests = efficiencies.shape[1]
    nbr_contrasts = contrasts.shape[0]

    # Distribution for each contrast
    eff_rep = np.zeros(efficiencies.shape)
    for c in range(nbr_contrasts):
        for i in range(eff_rep.shape[1]):
            eff_rep[c, i] = np.sum(efficiencies[c, :] < efficiencies[c, i])
    eff_rep = eff_rep / nbr_tests

    # Threshold the distribution and count intersections
    th_v = np.arange(1.0, 0., step=-0.01)
    nbr_good_tests = np.zeros(th_v.shape)
    i_best = -1
    for (i, th) in enumerate(th_v):
        bin_eff_rep = eff_rep >= th
        intersection = np.sum(bin_eff_rep, axis=0) == nbr_contrasts
        nbr_good_tests[i] = np.sum(intersection)
        if nbr_good_tests[i] >= n:
            # take the last to maximise the efficiency of the first contrast
            i_best = np.argwhere(intersection)[-n:].flatten()
            break

    np.save(op.join(params['output_path'], output_file), i_best)
    return


def find_avg_designs(params_path, params_file="params.p", efficiencies_file="efficiencies.npy", output_file="avgs.npy",
                     n=1, verbose=False):
    """Return one or several index(es) of designs close to the average efficiency for all contrasts.

    :param params_path: Path of the directory that contains the parameters file (by default, this is the output dir.).
    :param params_file: (optional) Parameters file name: Default: params.p
    :param efficiencies_file: (optional) Efficiencies file name. Default: efficiencies.npy
    :param n: (optional) Number of indexes to return. Default: 1.
    :param verbose: (optional) Set to True to get more printed details of the running. Default: False.
    :type n: int
    :return: Nothing.
    """
    # Load data
    params = pickle.load(open(op.join(params_path, params_file), "rb"))
    contrasts = params['contrasts']
    efficiencies = np.load(op.join(params['output_path'], efficiencies_file))

    nbr_tests = efficiencies.shape[1]
    nbr_contrasts = contrasts.shape[0]

    # Distribution for each contrast
    eff_rep = np.zeros(efficiencies.shape)
    for c in range(nbr_contrasts):
        for i in range(eff_rep.shape[1]):
            eff_rep[c, i] = np.sum(efficiencies[c, :] < efficiencies[c, i])
    eff_rep = eff_rep / nbr_tests

    # Threshold the distribution and count intersections
    th_h_v = np.arange(0.51, 1.00, step=+0.01)
    th_l_v = np.arange(0.49, 0., step=-0.01)
    nbr_good_tests = np.zeros(th_h_v.shape)
    i_best = -1
    for (i, th_h) in enumerate(th_h_v):
        th_l = th_l_v[i]
        bin_eff_rep = (eff_rep >= th_l) * (eff_rep <= th_h)
        intersection = np.sum(bin_eff_rep, axis=0) == nbr_contrasts
        nbr_good_tests[i] = np.sum(intersection)
        if nbr_good_tests[i] >= n:
            if verbose:
                print("Low threshold: {}\nHight threshold: {}".format(th_l, th_h))
            # take the last to maximise the efficiency of the first contrast
            i_best = np.argwhere(intersection)[-n:].flatten()
            break

    np.save(op.join(params['output_path'], output_file), i_best)
    return


if __name__ == "__main__":
    function = sys.argv[1]

    if function == "find_best_designs":
        find_best_designs(sys.argv[2], n=int(sys.argv[3]))

    elif function == "find_avg_designs":
        find_avg_designs(sys.argv[2], n=int(sys.argv[3]), verbose=True)

    else:
        warnings.warn("Unrecognized function '{}'".format(function))
