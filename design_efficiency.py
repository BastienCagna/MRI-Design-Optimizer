#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    MRI Design Efficiency
    =====================

    This module create a large set of design and compute efficiencies for each design and contrasts

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

"""

import numpy as np
import pickle
import sys
import time
from scipy import signal
from nistats.design_matrix import make_design_matrix, plot_design_matrix
import warnings
import pandas as pd

import matplotlib.pyplot as plt


class ConvergenceError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def design_matrix(design, tr, last_iti=2.0):
    """Construct a design matrix using given parameter and nistat package.

    :param tr: Repetition time (in seconds)
    :return: The design matrix given by nistats
    """
    # frame times
    total_duration = design['onset'][-1] + design['duration'][-1] + design['ITI'][-1]
    n_scans = np.ceil(total_duration/tr)
    frame_times = np.arange(n_scans) * tr

    # TODO: check what does the duration vector on the design matrix (does make change with with single event ? =
    # short block)
    x = make_design_matrix(frame_times, pd.DataFrame(design), drift_model='blank')
    return x


def design_to_ndarray(x, cond_names):
    # x is a dict with a key for each regressor. each element of x is also a dict whitin each key correspond to a
    # sample time. the output of the function is just a dict that contain the time series of each regressor (no time
    # values)
    x = x.to_dict()
    t_vect = np.sort(list(x['constant'].keys()))

    dm = []
    # For each regressor
    for cond in np.unique(cond_names):
        vals = []
        # Keep just the signal's value for each sample
        for t in t_vect:
            vals.append(x[cond][t])
        dm.append(vals)

    vals = []
    # Keep just the signal's value for each sample
    for t in t_vect:
        vals.append(x['constant'][t])
    dm.append(vals)

    return np.array(dm)


def efficiency(xmatrix, c):
    """Compute the efficiency of a design. The formula is based of the works of Friston K & al. (1999).

    :param xmatrix: Design matrix (created by nistats)
    :param c: Contrast. Note that last coeffciencient of the contrast vector corresponds to the intercept and must be
              set to 0.
    :return: float efficiency value of the design using the given contrast.
    """
    xcov = np.dot(xmatrix.T, xmatrix)
    ixcov = np.linalg.inv(xcov)
    e = 1 / np.dot( np.dot(c, ixcov), c.T)
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

    nbr_events = int(np.sum(count_by_cond_orig))
    cond_seqs = np.zeros((nbr_designs, nbr_events), dtype=int)
    cond_grps = np.zeros((nbr_designs, nbr_events), dtype=int)

    t = time.time()
    tentatives = 0
    i = 0
    while i < nbr_designs:
        if verbose is True:
            if np.mod(i, nbr_designs/100) == 0:
                print("[{:.3f}s] {:2.0f}%".format(time.time()-t, 100*i/nbr_designs))

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
        if cond_seqs.__contains__(seq):
            # If it is, try again
            tentatives + 1
            # If too many try have been performed, exit
            if tentatives > iter_max:
                raise ConvergenceError("Impossible to create new different design in {} iterations.".format(iter_max))
        else:
            cond_seqs[i] = seq.T
            cond_grps[i] = seq_grp
            i += 1

    return cond_seqs, cond_grps


# def generate_designs(params_path, params_file="params.p", designs_file="designs.p", start_at=2.0, verbose=False):
def generate_designs(nbr_seqs, cond_counts, filenames, durations, cond_of_files, cond_names, cond_groups, group_names,
                     tmp, tmn, iti_orig, start_time, question_dur=0, verbose=False):
    """Create random designs that follow the configuration paramaters (nbr of designs, transition matrix, tr ...etc).

    :param start_time: (optional) Time before the first onset (in seconds). Default: 2.0 seconds.
    :param verbose: (optional) Set to True to get more printed details of the running. Default: False.
    :return: a array of pandas' dataframe containing all the designs.
    """

    nbr_events = int(np.sum(cond_counts))

    # Get condtions order of all desgins
    if verbose:
        print('Generate conditions orders')
    conds, groups = generate_sequence(cond_counts, cond_groups, nbr_seqs, tmp, tmn, verbose=verbose)

    # Get file order of each designs and so get duration order also (will be used to generate design matrix)
    if verbose:
        print("Creating file list of each design")
    files_orders = []
    durations_tab = []
    for i, seq in enumerate(conds):
        # At the beginning, no files are already used
        used_files = np.zeros((nbr_events,))

        durations_row = []
        file_order = []
        for cond in seq:
            i_files = np.where((used_files == 0) * (cond_of_files == cond))[0]
            i_file = np.random.choice(i_files)
            file_order.append(filenames[i_file])
            used_files[i_file] = 1
            durations_row.append(durations[i_file])

        durations_tab.append(durations_row)
        files_orders.append(file_order)

    # Create a new onset vector by drawing new event independently of the condition index
    if verbose:
        print("Generating ITI orders")
    onsets = np.zeros((nbr_seqs, nbr_events))
    isi_maxs = np.zeros((nbr_seqs,))
    itis = np.zeros((nbr_seqs, nbr_events))
    for i in range(nbr_seqs):
        iti_indexes = np.random.permutation(nbr_events)
        iti_list = iti_orig[iti_indexes]
        itis[i] = iti_list

        onsets[i, 0] = start_time
        for j in range(1, nbr_events):
            # Next onset  = previosu onset + previous stimulus duration + ITI
            onsets[i, j] = onsets[i, j-1] + float(durations_tab[i][j-1]) + iti_list[j-1] + question_dur

        # Find maximal isi (for the filtering of design matrix)
        # FIXME: Interval between last onset and the end is not taken in account. Is it a problem?
        isi_v = onsets[i, 1:] - onsets[i, :-1]
        isi_maxs[i] = np.max(isi_v)

    # Set the paradigm as nistat does in a pandas' DataFrame
    designs = []
    for i in range(nbr_seqs):
        trial_types = np.array([cond_names[j] for j in conds[i]])
        trial_group = np.array([group_names[j-1] for j in groups[i]])
        design = {"onset": onsets[i], "trial_type": trial_types, "trial_group": trial_group, "type_idx": conds[i],
                  "files": files_orders[i], "duration": durations_tab[i], "ITI": itis[i]}
        designs.append(design)

    return designs, isi_maxs


def filtered_design_matrix(tr, design, isi_max, nf=6, fc1=1/120):
    # Construct the proper filter
    fs = 1 / tr
    w1 = 2 * fc1 / fs
    fc2 = 1 / isi_max
    w2 = 2 * fc2 / fs
    if w2 > 1:
        warnings.warn("w2 was automatically fixed to 1.")
        w2 = 1
    b, a = signal.iirfilter(nf, [w1, w2], rs=80, rp=0, btype='band', analog=False, ftype='butter')

    X = design_matrix(design, tr)

    # Filter each columns to compute efficiency only on the bold signal bandwidth
    for c in X.keys():
        # If filter is not doing good job, test him in tools/filter_benchmark.py
        # If ISI are too long, you might need to reduce the filter order.
        X[c] = signal.filtfilt(b, a, X[c])

    return X


def compute_efficiencies(tr, designs, contrasts, isi_maxs, nf=6, fc1=1/120, verbose=False):
    """Compute efficiencies of each designs and each contrasts.

    Each regressor of the design matrix (each one corresponding to a condition) is filtered by a butterwotrh filter and
    then the efficiency of the resulting matrix is computed.

    :param nf: (optional) Order of the butterworth filter. Default: 9.
    :param fc1: (optional) High pass cutting frequency of the filter (in Hertz). Default: 1/120 Hz.
    :param verbose: (optional) Set to True to get more printed details of the running. Default: False.
    :return:
    """
    nbr_tests = len(designs)

    nbr_contrasts = contrasts.shape[0]

    efficiencies = np.zeros((nbr_contrasts, nbr_tests))

    t = time.time()
    for (i, c) in enumerate(contrasts):
        print("[{:.3f}s] contrast {}/{}:".format(time.time() - t, i + 1, nbr_contrasts), end="   ")
        for k, design in enumerate(designs):
            # compute efficiency of all examples
            if verbose and np.mod(k, nbr_tests / 10) == 0:
                print("{:2.0f}%".format((100*k/nbr_tests)), end="  ")

            X = filtered_design_matrix(tr, design, isi_maxs[k], fc1=fc1, nf=nf)

            # Compute efficiency of this design for this contrast
            efficiencies[i, k] = efficiency(X, c)
        print("")

    return efficiencies


if __name__ == "__main__":
    function = sys.argv[1]

    if function == "generate_designs":
        params = pickle.load(open(sys.argv[2], "rb"))
        designs, isi_maxs = generate_designs(params['nbr_designs'], params['cond_counts'], params['files_list'],
                                   params['files_duration'], params['cond_of_files'], params['cond_groups'],
                                   params['TMp'], params['TMn'], verbose=True)
        np.save(sys.argv[3], designs)
    elif function == 'compute_efficiencies':
        compute_efficiencies(sys.argv[2], verbose=True)

    else:
        warnings.warn("Unrecognized function '{}'".format(function))
