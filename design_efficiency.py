import numpy as np
import os.path as op
import pickle
import sys
import time
from scipy import signal
from nistats.design_matrix import make_design_matrix
import warnings
import pandas as pd
import scipy.io as io
import scipy.io.wavfile as wf


class ConvergenceError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def create_soa_file():
    data = io.loadmat("/hpc/banco/bastien.c/localizer.mat")
    onsets = np.concatenate(data['onsets'].flatten()).flatten()
    onsets = np.sort(onsets)
    SOAs = onsets[1:] - onsets[:-1]

    last_onset = onsets[-1]

    nbr_soas = len(SOAs)
    nbr_manquants = 144 - nbr_soas

    start_at = np.mean(SOAs)

    SOAs = np.insert(SOAs, 0, start_at)

    np.save("/hpc/banco/bastien.c/data/optim/VL/banks/SOAs", SOAs)

    # plt.hist(SOAs, 20)
    # plt.title("Histogram of ISIs (or SOAs)")
    # plt.xlabel("ISI (s)")
    # plt.show()


def design_matrix(tr, conditions, onsets, final_isi):
    # frame times
    total_duration = onsets[-1] + final_isi
    n_scans = np.ceil(total_duration/tr)
    frame_times = np.arange(n_scans) * tr

    # event-related design matrix
    paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets})

    X = make_design_matrix(frame_times, paradigm, drift_model='blank')

    return X


def efficiency(xmatrix, c):
    xcov = np.dot(xmatrix.T, xmatrix)
    ixcov = np.linalg.inv(xcov)
    e = 1 / np.dot( np.dot(c, ixcov) , c.T )
    return e


def find_in_tmp(tmp, needed):
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
    """
    Generate nbr_seq sequences of events.
    :param count_by_cond:
    :param groups:
    :param nbr_designs:
    :param tmp: Np * Nh matrix describing all possible sequence composed of Nh events.
    :param tmn: Np * Nj matrix giving probabilities of new event type depending of the previous sequence.
    :param iter_max:
    :return:
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


# TODO: Completer la doc
def write_parameters_file(conditions_names, cond_of_files, groups, contrasts, contrast_names, files_list, files_path,
                          ITI_file, nbr_designs, tmp, tmn, tr, output_path, output_file="params.p"):
    """
    Create the design configuration file containning parameters that's will be used by the pipeline.

    :param output_file:
    :param conditions_names:
    :param cond_of_files:
    :param groups:
    :param contrasts:
    :param contrast_names:
    :param files_list:
    :param files_path:
    :param ITI_file:
    :param nbr_designs:
    :param tmp:
    :param tmn:
    :param tr:
    :param output_path:
    :return:
    """
    # TODO: amelioration: ajouter plus de controls

    # Count for each condition, how many times she's appearing
    cond_of_files = np.array(cond_of_files)
    ucond = np.unique(cond_of_files)
    nbr_cond = len(ucond)
    count_by_cond = np.zeros((nbr_cond,))
    for cond in ucond:
        count_by_cond[cond] = np.sum(cond_of_files==cond)

    # Durations
    durations = []
    for file in files_list:
        fs, x = wf.read(op.join(files_path, file))
        durations.append(len(x)/float(fs))
    durations = np.array(durations)

    # Save conditions and onsets
    params = {
        'cond_names': conditions_names,
        'cond_counts': count_by_cond,
        'cond_of_files': cond_of_files,
        'cond_groups': groups,
        'contrasts': contrasts,
        'constrast_names': contrast_names,
        'files_duration': durations,
        'files_list': files_list,
        'files_path': files_path,
        'ITI_file': ITI_file,
        'nbr_designs': nbr_designs,
        'TMp': tmp,
        'TMn': tmn,
        'tr': tr,
        'output_path': output_path
    }
    pickle.dump(params, open(op.join(output_path, output_file), "wb"))

    return True


def generate_designs(params_path, params_file="params.p", designs_file="designs.p", start_at=2.0, postStimTime=0,
                     verbose=False):
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


def compute_efficiencies(param_path, params_file="params.p", designs_file="designs.p", output_file="efficiencies.npy",
                         nf=9, verbose=False):
    # Read parameters
    params = pickle.load(open(op.join(param_path, params_file), "rb"))
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
    fc1 = 1 / 120
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


if __name__ == "__main__":
    function = sys.argv[1]

    # if function == "randomized_onsets":
    #     """ example: python /hpc/banco/bastien.c/python/designer/design_efficiency.py randomized_onsets
    #     /hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/params.p
    #     /hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/designs.npy """
    #     randomized_onsets(sys.argv[2], sys.argv[3])
    #
    # elif function == "randomized_SOA_order":
    #     randomized_soa_order(sys.argv[2], sys.argv[3], sys.argv[4])
    #
    # elif function == "randomized_ISI_order":
    #     if len(sys.argv) == 6:
    #         randomized_isi_order(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
    #     else:
    #         randomized_isi_order(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]),
    #                              postStimTime=float(sys.argv[6]))
    # elif function == "voice_calib_order":
    #         voice_calib_order(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
    # elif function == "voice_calib_order2":
    #         voice_calib_order2(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
    if function == "generate_designs":
        generate_designs(sys.argv[2], verbose=True)

    elif function == 'compute_efficiencies':
        compute_efficiencies(sys.argv[2], verbose=True)

    else:
        warnings.warn("Unrecognized function '{}'".format(function))
