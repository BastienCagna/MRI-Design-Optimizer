#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    MRI Design Efficiency
    =====================

    This module create a large set of design and compute efficiencies for each design and contrasts

    Steps of this pipeline require that you previously created a parameters file using create_parameters_file.py

"""

import numpy as np
import pandas as pd
import pickle
import sys
import time
import warnings
from scipy import signal
from nistats.design_matrix import make_design_matrix


class ConvergenceError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def design_matrix(design, tr):
    """Construct a design matrix using given parameter and nistat package.

    :param design: Design (dictionnary)
    :param tr: Repetition time (in seconds)
    :return: The design matrix given by nistats
    """
    # The total duration is equal to the last onset + the last stimulus duration + the last ITI
    total_duration = design['onset'][-1] + design['duration'][-1] + design['ITI'][-1]
    n_scans = np.ceil(total_duration/tr)
    frame_times = np.arange(n_scans) * tr

    # TODO: print to remove
    design = pd.DataFrame(design)
    # print("\nNew design:")
    # for k in design.keys():
    #     print("   {}: {}".format(k, len(design[k])))
    # print("   frame_times: {}".format(n_scans))

    # Create the design matrix using nistats and with any additional regressors (as this is not real data)

    # design.to_csv("/hpc/banco/bastien.c/data/optim/calibrator_iti/designs_iti_0_0/"
    #               "bad_design.csv")
    x = make_design_matrix(frame_times, design, drift_model='blank')

    return x


def efficiency(xmatrix, c):
    """Compute the efficiency of a design. The formula is based of the works of Friston K & al. (1999).

    :param xmatrix: Design matrix (created by nistats)
    :param c: Contrast. Note that last coeffciencient of the contrast vector corresponds to the intercept and must be
              set to 0.
    :return: Float efficiency value of the design using the given contrast.
    """
    xcov = np.dot(xmatrix.T, xmatrix)
    ixcov = np.linalg.inv(xcov)
    e = 1 / np.dot(np.dot(c, ixcov), c.T)
    return e


def find_in_tmp(tmp, needed):
    """Search the needed row in the TMp array.
    If the row is in the array, the row index is returned if not, -1 is returned.

    :param tmp: Transition Matrix previous as defined by Hanson (2015). Past conditions of the design sequence.
    :param needed: Needed past conditions vector.
    :return: The row index of needed in tmp or -1.
    """
    for i, item in enumerate(tmp):
        if np.sum(np.array(item == needed)) == tmp.shape[1]:
            n_required = item.shape[0]
            n = 0
            for j in range(n_required):
                if item[j] == needed[j]:
                    n += 1
                else:
                    break
            if n == n_required:
                return i
    return -1


def generate_sequence(nbr_events, count_by_cond_orig, groups, cond_names, nbr_designs, tmp, tmn,
                      iter_max=40, verbose=False):
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
    # Nbr of previous cases
    nh = tmp.shape[1]
    # List of conditions groups
    unique_grp = np.unique(groups)
    nbr_groups = len(unique_grp)

    # Event count for each group of conditions
    count_by_grp_orig = []
    conds_of_grp = []
    for grp in unique_grp:
        # Each row of conds_of_grp will contains the list of conditions names associated to this groups index
        cond_idx = np.argwhere(np.array(groups) == grp).flatten()
        cond_names_in_grp = []
        for c_idx in cond_idx:
            cond_names_in_grp.append(cond_names[c_idx])
        conds_of_grp.append(cond_names)

        # Sum every conditions count for each group
        sum = 0
        for i, cond_grp in enumerate(groups):
            if cond_grp == grp:
                sum += count_by_cond_orig[cond_names[i]]
        count_by_grp_orig.append(sum)

    # For each group, compute probs of each conditions
    probas = []
    for g in range(nbr_groups):
        s = 0
        for cond_name in conds_of_grp[g]:
            s += count_by_cond_orig[cond_name]
        p_v = []
        for cond_name in conds_of_grp[g]:
            p_v.append(count_by_cond_orig[cond_name] / s)
        probas.append(p_v)

    # cond_seqs = np.zeros((nbr_designs, nbr_events), dtype=int)
    cond_seqs = []
    cond_grps = np.zeros((nbr_designs, nbr_events), dtype=int)

    t = time.time()
    for i in range(nbr_designs):
        if verbose is True:
            if np.mod(i, nbr_designs/100) == 0:
                print("[{:.3f}s] {:2.0f}%".format(time.time()-t, 100*i/nbr_designs))

        # Try several time to get a design
        for k_test in range(iter_max):
            if k_test == iter_max-1:
                raise ConvergenceError("Cannot reah to build good design.z")

            # Vector of events
            seq = []
            # Vector of event's groups
            seq_grp = -1 * np.ones((nbr_events,), dtype=int)

            # Initialisation
            # drawn at random Nh first event type until the generated sequence is in TMp
            it = 0
            # Array used to count event's appearition
            count_by_grp = np.array(count_by_grp_orig)
            while find_in_tmp(tmp, seq_grp[:nh]) == -1:
                if it >= iter_max:
                    raise ConvergenceError("Design creation failed to generate initial group "
                                           "sequence.")
                for j in range(nh):
                    for it2 in range(iter_max):
                        grp = np.random.choice(unique_grp)
                        if count_by_grp[grp-1] > 0:
                            seq_grp[j] = grp
                            count_by_grp[grp-1] -= 1
                            break
                    if seq_grp[j] == -1:
                        raise ConvergenceError("Cannot initialise groups sequence.")

                # Re-init this count before start a new group giving out
                count_by_grp = np.array(count_by_grp_orig)
                it += 1
            count_by_grp_after_init = np.copy(count_by_grp)

            # 2 - Generate the sequence using TMp and TMn until there is anought events
            # If the sequence can not be finished, try several time
            for k in range(iter_max):
                seq_grp[nh:] = -1 * np.ones((nbr_events-nh,))
                count_by_grp = np.copy(count_by_grp_after_init)
                seq_completed = True
                j = nh
                # Continue the sequence
                while j < nbr_events:
                    # i_prev = np.argwhere(seq_grp[j-Nh:j] == tmp)[0, 0]
                    i_prev = find_in_tmp(tmp, seq_grp[j-nh:j])

                    # Get new group in the sequence using TMn's probabilities
                    for it2 in range(iter_max):
                        grp = np.random.choice(unique_grp, p=tmn[i_prev])
                        # If the count credit for this group is more than 0,
                        # use it otherwise try a new choice
                        if count_by_grp[grp-1] > 0:
                            seq_grp[j] = grp
                            count_by_grp[grp-1] -= 1
                            break

                    # If no choice succed to continue the sequence, stop here and restart after
                    # the init.
                    if seq_grp[j] == -1:
                        seq_completed = False
                        break
                    j += 1

                if seq_completed is True:
                    break
                # Else: try one more time

            # if group sequence is ok, poursue, else try a new design creation from scratch
            if seq_completed is True and len(seq_grp) == nbr_events:
                seq_grp = np.array(seq_grp)

                # If the sequence can not be finished, try several time
                for k1 in range(int(iter_max/10)):
                    seq_completed = True
                    count_by_cond = count_by_cond_orig.copy()
                    # Try to assign a condition to each event (corresponding to the suited group)
                    for j in range(nbr_events):
                        for it in range(iter_max):
                            cond = np.random.choice(conds_of_grp[seq_grp[j]-1], p=probas[(seq_grp[j]-1)])
                            # If the choosen conditinons can be used, go ahead, else try a new
                            # choice
                            if count_by_cond[cond] > 0:
                                seq.append(cond)
                                count_by_cond[cond] -= 1
                                break
                        # If event of index j was not added, the sequence can not be completed
                        if len(seq) <= j:
                            seq_completed = False
                            break

                    # if all event reveived a condition
                    if seq_completed is True and len(seq) == nbr_events and not \
                            cond_seqs.__contains__(seq):
                        cond_seqs.append(seq)
                        cond_grps[i] = seq_grp
                        seq_completed = "sequence_added"
                        break

            if seq_completed == "sequence_added":
                # Go to next
                break

    return cond_seqs, cond_grps


def generate_designs(nbr_seqs, nbr_events, cond_counts, filenames, durations, cond_of_files, cond_names, cond_groups,
                     group_names,
                     tmp, tmn, iti_orig, start_time, question_dur=0, verbose=False):
    """
    Create random designs that follow the configuration paramaters (nbr of designs, transition matrix, tr ...etc).

    :param nbr_seqs:
    :param nbr_events: Number of stimulation onsets (int)
    :param cond_counts: Number of stimulus for each cond (dictionnary)
    :param filenames:
    :param durations:
    :param cond_of_files:
    :param cond_names:
    :param cond_groups:
    :param group_names:
    :param tmp: Previous combinaison matrix of the transition table
    :param tmn: Next step matrix of the the transition table
    :param iti_orig: First ITI duration (before the first stimulation)
    :param start_time: (optional) Time before the first onset (in seconds). Default: 2.0 seconds.
    :param question_dur: (optional)Question duration (in seconds). Default: 0.
    :param verbose: (optional) Set to True to get more printed details of the running. Default: False.
    :return: An array containing all the designs as dataframes.
    """

    # Get conditions order of all desgins
    if verbose:
        print('Generate conditions orders')
    conds, groups = generate_sequence(nbr_events, cond_counts, cond_groups, cond_names, nbr_seqs, tmp, tmn,
                                      verbose=verbose)

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
        trial_types = np.array(conds[i])#np.array([cond_names[j] for j in conds[i]])
        trial_group = np.array([group_names[j-1] for j in groups[i]])
        design = {"onset": onsets[i], "trial_type": trial_types, "trial_group": trial_group,
                  "files": files_orders[i], "duration": durations_tab[i], "ITI": itis[i]}
        designs.append(design)

    return designs, isi_maxs


def filtered_design_matrix(tr, design, isi_max, nf=6, fc1=1/120, filt_type='band'):
    # Construct the proper filter
    fs = 1 / tr
    w1 = 2 * fc1 / fs
    fc2 = 1 / isi_max
    w2 = 2 * fc2 / fs
    if w2 > 1:
        warnings.warn("w2 was automatically fixed to 1.")
        w2 = 1

    if filt_type == 'band':
        b, a = signal.iirfilter(nf, [w1, w2], rs=80, rp=0, btype='band', analog=False, ftype='butter')
    elif filt_type == 'highpass':
        b, a = signal.iirfilter(nf, [w1], rs=80, rp=0, btype='highpass', analog=False, ftype='butter')
    else:
        b = None
        a = None

    d_matrix = design_matrix(design, tr)

    if filt_type == 'pass':
        return d_matrix

    # Filter each columns to compute efficiency only on the bold signal bandwidth
    for c in d_matrix.keys():
        # If filter is not doing good job, test him in tools/filter_benchmark.py
        # If ISI are too long, you might need to reduce the filter order.
        d_matrix[c] = signal.filtfilt(b, a, d_matrix[c])

    return d_matrix


def compute_efficiencies(tr, designs, contrasts, isi_maxs, nf=6, fc1=1/120, verbose=False):
    """Compute efficiencies of each designs and each contrasts.

    Each regressor of the design matrix (each one corresponding to a condition) is filtered by a butterwotrh filter and
    then the efficiency of the resulting matrix is computed.

    :param tr:
    :param designs:
    :param contrasts:
    :param isi_maxs:
    :param nf: (optional) Order of the butterworth filter. Default: 9.
    :param fc1: (optional) High pass cutting frequency of the filter (in Hertz). Default: 1/120 Hz.
    :param verbose: (optional) Set to True to get more printed details of the running. Default: False.
    :return:
    """
    nbr_tests = len(designs)
    nbr_contrasts = contrasts.shape[0]
    efficiencies = np.zeros((nbr_contrasts, nbr_tests))

    t = time.time()
    total_prc = 0
    for (i, c) in enumerate(contrasts):
        if verbose:
            print("[{:.3f}s] contrast {}/{}".format(time.time() - t,
                                                    i + 1, nbr_contrasts))
        for k, design in enumerate(designs):
            # compute efficiency of all examples
            if verbose and np.mod(k, nbr_tests / 100) == 0:
                prc = int(100 * k / nbr_tests)
                print("[{:.3f}s] {}%  {:.02f}%".format(time.time() - t, prc, (total_prc +
                                                                              prc/nbr_contrasts)))

            d_matrix = filtered_design_matrix(tr, design, isi_maxs[k], fc1=fc1, nf=nf)

            # Compute efficiency of this design for this contrast
            efficiencies[i, k] = efficiency(d_matrix, c)
        total_prc += 100/nbr_contrasts
    return efficiencies


if __name__ == "__main__":
    function = sys.argv[1]

    if function == "generate_designs":
        params = pickle.load(open(sys.argv[2], "rb"))
        # Todo: Update the parameters list to match the function call
        dsgns, isi_max_v = generate_designs(params['nbr_designs'], params['cond_counts'], params['files_list'],
                                            params['files_duration'], params['cond_of_files'], params['cond_groups'],
                                            params['TMp'], params['TMn'], verbose=True)
        pickle.dump({'design': dsgns, 'isi_maxs': isi_max_v}, open(sys.argv[3], "wb"))

    elif function == 'compute_efficiencies':
        # Todo: change the following line to match the new function signature
        compute_efficiencies(sys.argv[2], verbose=True)

    else:
        warnings.warn("Unrecognized function '{}'".format(function))
