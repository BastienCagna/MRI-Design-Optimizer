#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Desgins Selection
    =================

"""

import sys
import numpy as np
import pickle
import warnings


def find_best_designs(efficiencies, contrasts, n=1):
    """Return one or several index(es) of designs that provide jointly best efficiencies for all contrasts.

    :param efficiencies: Array of efficiencies. Each row correspond to a desing. Each column correspond to a contrast.
    :param contrasts: Array of contrasts.
    :param n: (optional) Number of indexes to return. Default: 1.
    :return: Numpy array of index(es) of best design(s).
    """

    nbr_tests = efficiencies.shape[1]
    nbr_contrasts = contrasts.shape[0]

    # Distribution for each contrast
    eff_rep = np.zeros(efficiencies.shape)
    for c in range(nbr_contrasts):
        for i in range(eff_rep.shape[1]):
            eff_rep[c, i] = np.sum(np.array(efficiencies[c, :] < efficiencies[c, i]))
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

    return i_best


def find_avg_designs(efficiencies, contrasts, n=1):
    """Return one or several index(es) of designs close to the average efficiency for all contrasts.

    :param efficiencies: Array of efficiencies. Each row correspond to a desing. Each column correspond to a contrast.
    :param contrasts: Array of contrasts.
    :param n: (optional) Number of indexes to return. Default: 1.
    :return: Numpy array of index(es) of selected design(s).
    """
    nbr_tests = efficiencies.shape[1]
    nbr_contrasts = contrasts.shape[0]

    # Distribution for each contrast
    eff_rep = np.zeros(efficiencies.shape)
    for c in range(nbr_contrasts):
        for i in range(eff_rep.shape[1]):
            eff_rep[c, i] = np.sum(np.array(efficiencies[c, :] < efficiencies[c, i]))
    eff_rep = eff_rep / nbr_tests

    # Threshold the distribution and count intersections
    th_h_v = np.arange(0.51, 1.00, step=+0.01)
    th_l_v = np.arange(0.49, 0., step=-0.01)
    nbr_good_tests = np.zeros(th_h_v.shape)
    i_avg = -1
    for (i, th_h) in enumerate(th_h_v):
        th_l = th_l_v[i]
        bin_eff_rep = (eff_rep >= th_l) * (eff_rep <= th_h)
        intersection = np.sum(bin_eff_rep, axis=0) == nbr_contrasts
        nbr_good_tests[i] = np.sum(intersection)
        if nbr_good_tests[i] >= n:
            print("Low threshold: {}\nHight threshold: {}".format(th_l, th_h))
            # take the last to maximise the efficiency of the first contrast
            i_avg = np.argwhere(intersection)[-n:].flatten()
            break

    return i_avg


def average_relative_efficiencies(efficiencies, contrasts):
    # Compute relative efficiencies of each design for each contrast
    print("Computing relative efficiencies...")
    relatives_efficiencies = np.zeros(efficiencies.shape)
    for i, c in enumerate(contrasts):
        print("\tContrasts {: 2d}/{}".format(i+1, len(contrasts)))
        sorted_eff = np.sort(efficiencies[i])
        arg_sorted_eff = np.argsort(efficiencies[i])

        cum_eff = np.cumsum(sorted_eff)
        relative_eff = cum_eff / np.repeat(cum_eff[-1:], cum_eff.shape[0])

        relatives_efficiencies[i] = relative_eff[arg_sorted_eff]

    # Compute the score of each design by averaging relative efficiency of all contrast
    print("Computing scores...")
    avg_eff = np.mean(relatives_efficiencies, axis=0)
    print("Done")
    return avg_eff


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Design selection")
        print("\nReturn one or several design(s) index(es) following there efficiencies over each contrasts.")
        print("\n*** Args ***")
        print("[1]  Algo. 'best': best designs 'avg': average designs. \n[2]  Parameters file")
        print("[3]  Efficiencies numpy file\n[4]  Number of design to select\n[5] Output file (.npy)\n\n")
        exit(0)

    function = sys.argv[1]
    params = pickle.load(open(sys.argv[2], "rb"))
    eff_file = np.load(sys.argv[3])

    if function == "best":
        n_sel = int(sys.argv[4])
        output_file = sys.argv[5]
        idxes = find_best_designs(eff_file, params['contrasts'], n_sel)
        np.save(output_file, idxes)

    elif function == "avg":
        n_sel = int(sys.argv[4])
        output_file = sys.argv[5]
        idxes, avg_efficiencies = find_avg_designs(eff_file, params['contrasts'], n_sel)
        np.save(output_file, idxes)

    elif function == "score":
        output_file = sys.argv[4]
        designs_score = average_relative_efficiencies(eff_file, params['contrasts'])
        np.save(output_file, designs_score)
        # # Return designs' indexes ordered by ascending score (average relative efficiency)
        # print("Sorting designs...")
        # designs_order = np.argsort(avg_eff)

    else:
        warnings.warn("Unrecognized function '{}'".format(function))
