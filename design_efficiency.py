#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    MRI Design Efficiency
    =====================

    This module is made to find more efficient design in a large number of possible designs.

    Steps of this pipeline require that you previously created a parameters file using create_parameters_file.py

    :Example
    1 - Create the designs set:
    $ python design_effeciency.py generate_designs /hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/

    2 - Compute corresponding efficiencies (for all designs and all contrasts):
    $ python design_effeciency.py compute_efficiencies /hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/

    3 - Find the best designs (only one or several, here the 9 best):
    $ python design_effeciency.py find_best_desings /hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/ 9

    4 - (optional) Find some 'average' designs (only one or several, here the 9 best):
    $ python design_effeciency.py find_avg_desings /hpc/banco/bastien.c/data/optim/identification/test_new_pipeline/ 9

    Todo:
    * Generalise the process to any type of stimulation files. This version can be used only with .wav file for audio
    stimulation.
"""

import numpy as np
import os.path as op
import pickle
import sys
import time
from scipy import signal
from nistats.design_matrix import make_design_matrix
import warnings
import pandas as pd


class ConvergenceError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def design_matrix(tr, conditions, onsets, final_isi):
    """Construct a design matrix using given parameter and nistat package.

    :param tr: Repetition time (in seconds)
    :param conditions: Vector giving condition of each stimulation of the scan as integer
    :param onsets: Vector of onsets of all stimuli of the scan (in seconds)
    :param final_isi: Last ISI (in seconds)
    :return: The design matrix given by nistats
    """
    # frame times
    total_duration = onsets[-1] + final_isi
    n_scans = np.ceil(total_duration/tr)
    frame_times = np.arange(n_scans) * tr

    # event-related design matrix
    paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets})

    x = make_design_matrix(frame_times, paradigm, drift_model='blank')
    return x


def efficiency(xmatrix, c):
    """Compute the efficiency of a design. The formula is based of the works of Friston K & al. (1999).

    :param xmatrix: Design matrix (created by nistats)
    :param c: Contrast. Note that last coeffciencient of the contrast vector corresponds to the intercept and must be
              set to 0.
    :return: float efficiency value of the design using the given contrast.
    """
    xcov = np.dot(xmatrix.T, xmatrix)
    ixcov = np.linalg.inv(xcov)
    e = 1 / np.dot( np.dot(c, ixcov) , c.T )
    return e


def find_in_tmp(tmp, needed):
    """Search the needed row in the TMp array.
    If the row is in the array, the row index is returned if not, -1 is returned.

    :param tmp: Transition Matrix previous as defined by Hanson (2015). Past conditions of the design sequence.
    :param needed: Needed past conditions vector.
    :return: The row index of needed in tmp or -1.
    """
    for i, item in enumerate(tmp):
        if np.sum(item == needed)==tmp.shape[1]:
            n_required = item.shape[0]
            n = 0
            for j in range(n_required):
                if item[j] == needed[j]:
                    n += 1
                else:
                    break
            if n==n_required:
                return i
    return -1


def generate_sequence(count_by_cond_orig, groups, nbr_designs, tmp, tmn, iter_max=1000, verbose=False):
    """Generate nbr_seq different sequences of events.

    Each sequence is composed of conditions indexes following the given count for
    each condition. Those conditions can be grouped and tmp and tmn give the probability to get next group in function
    of which groups have alreeady been drawed. See Design Efficiency, Hanson RN, Elsevier, 2015 for more explainations.

    :param count_by_cond_orig: Number of event for each condition.
    :param groups: Give the group index if each condition.
    :param nbr_designs: Number of needed designs.
    :param tmp: Np * Nh matrix describing all possible sequence composed of Nh events (Hanson, 2015).
    :param tmn: Np * Nj matrix giving probabilities of new event type depending of the previous sequence (Hanson, 2015).
    :param iter_max: (optional) Maximal number of trial of the different drawing loops of the algorithm. Default: 1000.
    :param verbose:  (optional) Set to True to get more printed details of the running. Default: False.
    :return: An array of conditions sequences. Each row corresponds to a design.
    """
    # Nbr of conditions
    Nj = len(count_by_cond_orig)
    # Nbr of previous cases
    Nh = tmp.shape[1]
    # List of conditions groups
    unique_grp = np.sort(np.unique(groups))

    count_by_grp_orig = []
    conds_of_grp = []
    for grp in unique_grp:
        conds_of_grp.append(np.argwhere(np.array(groups) == grp).flatten())
        count_by_grp_orig.append(np.sum(count_by_cond_orig * (groups == grp)))

    # For each group, compute probs of each conditions
    nbr_groups = len(unique_grp)
    probas = []
    for g in range(nbr_groups):
        s = np.sum(count_by_cond_orig[conds_of_grp[g]])
        probas.append(count_by_cond_orig[conds_of_grp[g]] / s)

    if Nj != tmn.shape[1]:
        warnings.warn("Conditions count doesn't match TMn's columns count.")
    # TODO: resolve the comparison problem
    # if unique_grp != np.arange(0, Nj):
        # warnings.warn("Groupes indexes are not from 0 to Nj-1.")

    nbr_events = int(np.sum(count_by_cond_orig))
    seqs = np.zeros((nbr_designs, nbr_events), dtype=int)

    t = time.time()
    tentatives = 0
    i = 0
    while i < nbr_designs:
        if verbose is True and np.mod(i, nbr_designs/10) == 0:
            print("%f - %d" % (time.time()-t, i))

        # Vector of events
        seq = -1 * np.ones((nbr_events, 1), dtype=int)
        # Vector giving the event's groups
        seq_grp = -1 * np.ones((nbr_events,), dtype=int)

        # 1 - Initialisation: drawn at random Nh first event type untill the generated sequence is in TMp
        it = 0
        while find_in_tmp(tmp, seq_grp[:Nh])==-1:
            # Array used to count event's appearition
            count_by_grp = np.array(count_by_grp_orig)
            if it >= iter_max:
                raise ConvergenceError("Maximal iteration number attained for the design {}. Cannot init the design."
                                       .format(i+1))
            for j in range(Nh):
                for it2 in range(iter_max):
                    grp = np.random.choice(unique_grp)
                    if count_by_grp[grp-1]>0:
                        seq_grp[j] = grp
                        count_by_grp[grp-1] -= 1
                        break
                if seq_grp[j] == -1:
                    raise ConvergenceError("Attribution of group was not successful (during initiation).")
            it += 1
        count_by_grp_after_init = np.copy(count_by_grp)

        # 2 - Generate the sequence using TMp and TMn until there is anought events
        # If the sequence can not be finished, try several time
        for k in range(iter_max):
            seq_grp[Nh:] = -1 * np.ones((nbr_events-Nh,))
            count_by_grp = np.copy(count_by_grp_after_init)
            seq_completed = True
            j = Nh
            # Continue the sequence
            while j < nbr_events:
                # i_prev = np.argwhere(seq_grp[j-Nh:j] == tmp)[0, 0]
                i_prev = find_in_tmp(tmp, seq_grp[j-Nh:j])

                # Get new group in the sequence using TMn's probabilities
                for it2 in range(iter_max):
                    grp = np.random.choice(unique_grp, p=tmn[i_prev])
                    # If the count credit for this group is more than 0, use it otherwise try a new choice
                    if count_by_grp[grp-1] > 0:
                        seq_grp[j] = grp
                        count_by_grp[grp-1] -= 1
                        break

                # If no choice successed to continue the sequence, stop here and restart after the init.
                if seq_grp[j] == -1:
                    seq_completed = False
                    break

                j += 1

            if seq_completed:
                break
        if seq_completed is False:
            raise ConvergenceError("The design didn't succeded to complete during groups attribution.")
        seq_grp = np.array(seq_grp)

        # If the sequence can not be finished, try several time
        for k1 in range(iter_max):
            seq_completed = True
            count_by_cond = np.copy(count_by_cond_orig)
            for j in range(nbr_events):
                for it in range(iter_max):
                    cond = np.random.choice(conds_of_grp[seq_grp[j]-1], p=probas[(seq_grp[j]-1)])
                    if count_by_cond[cond] > 0:
                        seq[j] = cond
                        count_by_cond[cond] -= 1
                        break
                if seq[j] == -1:
                    seq_completed = False
                    break
            if seq_completed:
                break

        if seq_completed is False:
            raise ConvergenceError("The design didn't succeded to complete during conditions attribution.")

        # Test if the sequence already exist
        if seqs.__contains__(seq):
            # If it is, try again
            tentatives + 1
            # If too many try have been performed, exit
            if tentatives > iter_max:
                raise ConvergenceError("Impossible to create new different design in {} iterations.".format(iter_max))
        else:
            seqs[i] = seq.T
            i += 1

    return seqs


def generate_designs(params_path, params_file="params.p", designs_file="designs.p", start_at=2.0, verbose=False):
    """Create random designs that follow the configuration paramaters (nbr of designs, transition matrix, tr ...etc).

    :param params_path: Path of the directory that contains the parameters file (by default output directory).
    :param params_file: (optional) Parameters file name. Default: params.p
    :param designs_file: (optional) Designs file name in the output directory. Default: designs.p
    :param start_at: (optional) Time before the first onset (in seconds). Default: 2.0 seconds.
    :param verbose: (optional) Set to True to get more printed details of the running. Default: False.
    :return:
    """
    params = pickle.load(open(op.join(params_path, params_file), "rb"))
    nbr_seqs = params['nbr_designs']
    cond_counts = params['cond_counts']
    filenames = params['files_list']
    durations = params['files_duration']
    cond_of_files = params['cond_of_files']

    nbr_events = int(np.sum(cond_counts))

    # Get condtions order of all desgins
    if verbose:
        print('Generate conditions orders')
    conds = generate_sequence(cond_counts, params['cond_groups'], nbr_seqs, params['TMp'], params['TMn'],
                              verbose=verbose)

    # Get file order of each designs and so get duration order also (will be used to generate design matrix)
    if verbose:
        print("Creating file list of each desing")
    files_orders = []
    durations_tab = []
    for i, seq in enumerate(conds):
        # At the beginning, no files are already used
        used_files = np.zeros((nbr_events,))

        durations_row = []
        file_order = []
        for cond in seq:
            i_files = np.where((used_files==0) * (cond_of_files==cond))[0]
            i_file = np.random.choice(i_files)
            file_order.append(filenames[i_file])
            used_files[i_file] = 1
            durations_row.append(durations[i_file])

        durations_tab.append(durations_row)
        files_orders.append(file_order)

    # np.save(op.join(output_path, "file_orders"), files_orders)
    # np.save(op.join(output_path, "durations_tab"), durations_tab)

    # Create a new onset vector by drawing new event independently of the condition index
    if verbose:
        print("Generating ITI orders")
    iti_orig = np.load(params['ITI_file'])
    onsets = np.zeros((nbr_seqs, nbr_events))
    isi_maxs = np.zeros((nbr_seqs,))
    for i in range(nbr_seqs):
        iti_indexes = np.random.permutation(nbr_events-1)
        iti_list = iti_orig[iti_indexes]

        onsets[i, 0] = start_at
        for j in range(1, nbr_events):
            onsets[i, j] = onsets[i, j-1] + float(durations_tab[i][j-1]) + iti_list[j-1] # .replace(',', '.')

        # Find maximal isi (for the filtering of design matrix)
        isi_v = onsets[i, 1:] - onsets[i, :-1]
        isi_maxs[i] = np.max(isi_v)

    # Save designs
    designs = {"onsets": onsets, "conditions": conds, "files": files_orders, "durations": durations_tab,
               'max_ISIs': isi_maxs}
    pickle.dump(designs, open(op.join(params['output_path'], designs_file), "wb"))
    if verbose:
        print("Designs have been saved to: " + op.join(params['output_path'], designs_file))
    # np.save(designs_file, [onsets, conds, files_orders, durations_tab])
    return


def compute_efficiencies(params_path, params_file="params.p", designs_file="designs.p", output_file="efficiencies.npy",
                         nf=9, fc1=1/120, verbose=False):
    """Compute efficiencies of each designs and each contrasts.

    Each regressor of the design matrix (each one corresponding to a condition) is filtered by a butterwotrh filter and
    then the efficiency of the resulting matrix is computed.

    :param params_path: Path of the directory that contains the parameters file (by default, this is the output dir.).
    :param params_file: (optional) Parameters file name: Default: params.p
    :param designs_file: (optional) Designs file name: Default: designs.p
    :param output_file: (optional) Efficiencies file name. Default: efficiencies.npy.
    :param nf: (optional) Order of the butterworth filter. Default: 9.
    :param fc1: (optional) High pass cutting frequency of the filter (in Hertz). Default: 1/120 Hz.
    :param verbose: (optional) Set to True to get more printed details of the running. Default: False.
    :return:
    """
    # Read parameters
    params = pickle.load(open(op.join(params_path, params_file), "rb"))
    output_path = params['output_path']
    tr = params['tr']
    nbr_tests = params['nbr_designs']
    contrasts = params['contrasts']

    # Read designs
    designs = pickle.load(open(op.join(output_path, designs_file), "rb"))
    onsets = designs['onsets']
    conditions = np.array(designs['conditions'], dtype=int)
    durations_tab = designs['durations']
    isi_maxs = designs['max_ISIs']

    nbr_contrasts = contrasts.shape[0]

    efficiencies = np.zeros((nbr_contrasts, nbr_tests))

    # Unvariants filter parameters
    fs = 1 / tr
    w1 = 2 * fc1 / fs

    t = time.time()
    for (i, c) in enumerate(contrasts):
        for k in range(nbr_tests):
            # Construct the proper filte
            fc2 = 1 / isi_maxs[k]
            w2 = 2 * fc2 / fs
            if w2 > 1:
                warnings.warn("w2 was automatically fixed to 1.")
                w2 = 1
            b, a = signal.iirfilter(nf, [w1, w2], rs=80, rp=0, btype='band', analog=False, ftype='butter')

            # compute efficiency of all examples
            if verbose and np.mod(k, nbr_tests / 10) == 0:
                print("%ds: contrast %d/%d - efficiency evaluation %d/%d" %
                      (time.time() - t, i + 1, nbr_contrasts, k + 1, nbr_tests))

            X = design_matrix(tr, conditions[k], onsets[k], final_isi=durations_tab[k, -1])

            # Filter each columns to compute efficiency only on the bold signal bandwidth
            for j in range(X.shape[1] - 1):
                X[j] = signal.filtfilt(b, a, X[j])

            # Compute efficiency of this design for this contrast
            efficiencies[i, k] = efficiency(X, c)

    np.save(op.join(output_path, output_file), efficiencies)
    if verbose:
        print("Efficiencies saved at: " + op.join(output_path, output_file))
    return


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
    """
        All steps of the pipeline can be directly runned separatly by command line.
    """

    function = sys.argv[1]

    if function == "generate_designs":
        generate_designs(sys.argv[2], verbose=True)

    elif function == 'compute_efficiencies':
        compute_efficiencies(sys.argv[2], verbose=True)

    elif function == "find_best_designs":
        find_best_designs(sys.argv[2], n=int(sys.argv[3]))

    elif function == "find_avg_designs":
        find_avg_designs(sys.argv[2], n=int(sys.argv[3]))

    else:
        warnings.warn("Unrecognized function '{}'".format(function))
