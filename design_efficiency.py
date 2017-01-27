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


# TODO: Make a function that give the design matrix
def design_matrix(tr, conditions, onsets, durations, final_soa):
    # frame times
    # TODO:   ----- /!\ total duration has changed !!! ----------
    total_duration = onsets[-1]  + final_soa #+ durations[conditions[-1]]
    n_scans = np.ceil(total_duration/tr)
    # print('Total duration: %ds, Numbers of scans: %d' % (total_duration, n_scans))
    frame_times = np.arange(n_scans) * tr

    # event-related design matrix
    paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets})

    X = make_design_matrix(frame_times, paradigm, drift_model='blank')

    return X


# TODO: Make a function that compute the desin matrix's efficiency
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


def generate_sequence(count_by_cond_orig, groups, nbr_designs, tmp, tmn, iter_max=1000):
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
        # print("%f - %d" % (time.time()-t, i))
        # Vector of events
        seq = -1 * np.ones((nbr_events,), dtype=int)
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

        # For each group, compute probs of each conditions
        nbr_groups = len(unique_grp)
        nbr_conds = Nj
        # TODO: there is a problem here
        probas = np.zeros((nbr_groups,))
        for g in range(nbr_groups):
            s = np.sum(count_by_cond_orig[conds_of_grp[g]])
            probas[g] = count_by_cond_orig[conds_of_grp[g]] / s

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
            if np.mod(i, 1000)==0:
                print("designs creation at {}s: {}/{} completed".format(time.time()-t, i+1, nbr_designs))
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
        durations.append(len(x)*fs)
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


# 2 - Create designs
# def randomized_onsets(params_file, designs_file):
#     params = pickle.load(open(params_file, "rb"))
#     nbr_seqs = params['nbr_designs']
#     nbr_events = np.sum(params['cond_counts'])
#     conds = generate_sequence(params['cond_counts'], params['cond_groups'], nbr_seqs, params['TMp'], params['TMn'])
#
#     # Create a new onset vector by drawing new event independently of the condition index
#     SOAs = np.random.random((nbr_seqs, nbr_events)) * (params['SOAmax'] - params['SOAmin']) + params['SOAmin']
#     onsets = np.zeros((nbr_seqs, nbr_events))
#     tril_inf = np.tril(np.ones((nbr_events, nbr_events)), 0)
#     for i in range(nbr_seqs):
#         onsets[i] = np.dot(tril_inf, SOAs[i])
#
#     np.save(designs_file, [onsets, conds])
#     return
#
#
# def randomized_soa_order(params_file, designs_file, SOAs_file, postStimTime=0):
#     params = pickle.load(open(params_file, "rb"))
#     nbr_seqs = params['nbr_designs']
#     nbr_events = int(np.sum(params['cond_counts']))
#     conds = generate_sequence(params['cond_counts'], params['cond_groups'], nbr_seqs, params['TMp'], params['TMn'])
#
#     # Create a new onset vector by drawing new event independently of the condition index
#     SOAs_orig = np.load(SOAs_file)
#     onsets = np.zeros((nbr_seqs, nbr_events))
#     tril_inf = np.tril(np.ones((nbr_events, nbr_events)), 0)
#     for i in range(nbr_seqs):
#         SOAs_indexes = np.random.permutation(nbr_events)
#         SOAs = SOAs_orig[SOAs_indexes]
#         onsets[i] = np.dot(tril_inf, SOAs)
#
#     np.save(designs_file, [onsets, conds])
#     return
#
#
# def randomized_isi_order(params_file, designs_file, isi_file, start_at=2.0, postStimTime=0):
#     params = pickle.load(open(params_file, "rb"))
#     nbr_seqs = params['nbr_designs']
#     nbr_events = int(np.sum(params['cond_counts']))
#     durations = params['cond_durations']
#     conds = generate_sequence(params['cond_counts'], params['cond_groups'], nbr_seqs, params['TMp'], params['TMn'])
#
#     # Create a new onset vector by drawing new event independently of the condition index
#     isi_orig = np.load(isi_file)
#     onsets = np.zeros((nbr_seqs, nbr_events))
#     for i in range(nbr_seqs):
#         isi_indexes = np.random.permutation(nbr_events-1)
#         isi_list = isi_orig[isi_indexes]
#
#         onsets[i, 0] = start_at
#         for j in range(1, nbr_events):
#             onsets[i, j] = onsets[i, j-1] + durations[conds[i, j-1]] + isi_list[j-1] + postStimTime
#
#     np.save(designs_file, [onsets, conds])
#     return
#
#
# def voice_calib_order(params_file, designs_file, start_at=2.0, postStimTime=0):
#     params = pickle.load(open(params_file, "rb"))
#     nbr_seqs = params['nbr_designs']
#     nbr_events = int(np.sum(params['cond_counts']))
#     cond_names = params['cond_names']
#     output_path = params['output_path']
#
#     infos = pd.read_csv("/hpc/banco/bastien.c/data/optim/VC/infos.csv")
#     filenames = infos['file name']
#     durations = infos['duration']
#     conditions = infos['condition']
#
#     file_indexes = []
#     conds = []
#
#     for i in range(nbr_seqs):
#         if np.mod(i, 500)==0:
#             print(i)
#         tirage = np.random.permutation(nbr_events)
#         file_indexes.append(tirage)
#         row_cond = []
#         for t in tirage:
#             file = filenames[t]
#             row_cond.append(np.where(conditions[t]==cond_names)[0][0])
#         conds.append(row_cond)
#
#
#     # conds = generate_sequence(params['cond_counts'], params['cond_groups'], nbr_seqs, params['TMp'], params['TMn'])
#     np.save(op.join(output_path, "conditions"), conds)
#
#     infos = pd.read_csv("/hpc/banco/bastien.c/data/optim/VC/infos.csv")
#     filenames = infos['file name']
#     durations = infos['duration']
#     conditions = infos['condition']
#
#     cond_nums = np.arange(len(params['cond_counts']))
#     conditions_nums = []
#     for c in conditions:
#         for i_tmp in range(len(params['cond_counts'])):
#             if cond_names[i_tmp] == c:
#                 i_c = i_tmp
#                 break
#
#         conditions_nums.append(cond_nums[i_c])
#     conditions_nums = np.array(conditions_nums)
#
#     files_orders = []
#     durations_tab = []
#     for i, seq in enumerate(conds):
#         file_order = []
#         used_files = np.zeros((nbr_events,))
#         durations_row = []
#         for cond in seq:
#             i_file = np.where((used_files==0) * (conditions_nums==cond))[0][0]
#
#             file_order.append(filenames[i_file])
#             used_files[i_file] = 1
#             durations_row.append(durations[i_file])
#         durations_tab.append([durations_row])
#         files_orders.append(file_order)
#     np.save(op.join(output_path, "file_orders"), files_orders)
#     np.save(op.join(output_path, "durations_tab"), durations_tab)
#     # durations_tab = np.array(durations)
#
#     # Create a new onset vector by drawing new event independently of the condition index
#     isi_orig = np.load("/hpc/banco/bastien.c/data/optim/VC/ITIs_const.npy")
#     np.save(op.join(output_path, "ITIs"), isi_orig)
#     onsets = np.zeros((nbr_seqs, nbr_events))
#     for i in range(nbr_seqs):
#         iti_indexes = np.random.permutation(nbr_events-1)
#         iti_list = isi_orig[iti_indexes]
#
#         onsets[i, 0] = start_at
#         for j in range(1, nbr_events):
#             onsets[i, j] = onsets[i, j-1] + float(durations_tab[i][0][j-1]) + iti_list[j-1] # .replace(',', '.')
#
#     np.save(designs_file, [onsets, conds])
#     return
#

def generate_designs(params_path, params_file="params.p", designs_file="designs.p", start_at=2.0, postStimTime=0):
    params = pickle.load(open(op.join(params_path, params_file), "rb"))
    nbr_seqs = params['nbr_designs']
    cond_counts = params['cond_counts']
    filenames = params['files_list']
    durations = params['files_duration']
    cond_of_files = params['cond_of_files']

    nbr_events = int(np.sum(cond_counts))

    # Get condtions order of all desgins
    conds = generate_sequence(cond_counts, params['cond_groups'], nbr_seqs, params['TMp'], params['TMn'])

    # Get file order of each designs and so get duration order also (will be used to generate design matrix)
    files_orders = []
    durations_tab = []
    for i, seq in enumerate(conds):
        # At the beginning, no files are already used
        used_files = np.zeros((nbr_events,))

        durations_row = []
        file_order = []
        for cond in seq:
            i_file = np.where((used_files==0) * (cond_of_files==cond))#[0][0]

            file_order.append(filenames[i_file])
            used_files[i_file] = 1
            durations_row.append(durations[i_file])

        durations_tab.append([durations_row])
        files_orders.append(file_order)

    # np.save(op.join(output_path, "file_orders"), files_orders)
    # np.save(op.join(output_path, "durations_tab"), durations_tab)

    # Create a new onset vector by drawing new event independently of the condition index
    iti_orig = np.load(params['ITI_file'])
    onsets = np.zeros((nbr_seqs, nbr_events))
    isi_maxs = np.zeros((nbr_seqs,))
    for i in range(nbr_seqs):
        iti_indexes = np.random.permutation(nbr_events-1)
        iti_list = iti_orig[iti_indexes]

        onsets[i, 0] = start_at
        for j in range(1, nbr_events):
            onsets[i, j] = onsets[i, j-1] + float(durations_tab[i][0][j-1]) + iti_list[j-1] # .replace(',', '.')

        # Find maximal isi (for the filtering of design matrix)
        isi_v = onsets[i, 1:] - onsets[i, :-1]
        isi_maxs[i] = np.max(isi_v)

    # Save designs
    designs = {"onsets": onsets, "conditions": conds, "files": files_orders, "durations": durations_tab,
               'max_ISIs': isi_maxs}
    pickle.dump(designs, op.join(params['output_path'], designs_file))
    # np.save(designs_file, [onsets, conds, files_orders, durations_tab])
    return


def voice_calib_order2(params_file, designs_file, start_at=2.0, postStimTime=0):
    params = pickle.load(open(params_file, "rb"))
    nbr_seqs = params['nbr_designs']
    nbr_events = int(np.sum(params['cond_counts']))
    cond_names = params['cond_names']
    output_path = params['output_path']

    conds = np.load(op.join(output_path, "conditions.npy"))

    infos = pd.read_csv("/hpc/banco/bastien.c/data/optim/VC/infos.csv")
    filenames = infos['file name']
    durations = infos['duration']
    conditions = infos['condition']

    # cond_nums = np.arange(len(params['cond_counts']))
    # conditions_nums = []
    # for c in conditions:
    #     for i_tmp in range(len(params['cond_counts'])):
    #         if cond_names[i_tmp] == c:
    #             i_c = i_tmp
    #             break
    #
    #     conditions_nums.append(cond_nums[i_c])
    # conditions_nums = np.array(conditions_nums)

    files_orders = np.load(op.join(output_path, "file_orders.npy"))
    # durations_tab = []
    t =time.time()
    # for i, seq in enumerate(conds):
    #     if np.mod(i, 100) == 0:
    #         print("{}: {}".format(time.time()-t, i))
    #     durations_row = []
    #     for file in files_orders[i]:
    #         i_file = np.where((filenames==file))[0][0]
    #         durations_row.append(durations[i_file])
    #     durations_tab.append([durations_row])
    durations_tab = np.load(op.join(output_path, "duratiosn_tab.npy"))

    # Create a new onset vector by drawing new event independently of the condition index
    # isi_orig = np.load("/hpc/banco/bastien.c/data/optim/VC/ITIs.npy")
    np.save(op.join(output_path, "ISIs"), isi_orig)
    onsets = np.zeros((nbr_seqs, nbr_events))
    for i in range(nbr_seqs):
        if np.mod(i, 100) == 0:
            print("onsets - {}: {}".format(time.time()-t, i))
        isi_indexes = np.random.permutation(nbr_events-1)
        isi_list = isi_orig[isi_indexes]

        onsets[i, 0] = start_at
        for j in range(1, nbr_events):
            onsets[i, j] = onsets[i, j-1] + float(durations_tab[i][0][j-1].replace(',', '.')) + isi_list[j-1] + postStimTime

    np.save(designs_file, [onsets, conds])
    return


# 3 - Compute efficiencies
def compute_efficiencies(param_filename, designs_file, output_file, nf=9):
    # Read parameters
    params = pickle.load(open(param_filename, "rb"))
    durations = params['cond_durations']
    tr = params['tr']
    nbr_tests = params['nbr_designs']
    contrasts = params['contrasts']

    # Read designs
    # [onsets, conditions] = np.load(designs_file)
    designs = pickle.load(designs_file)
    onsets = designs['onsets']
    conditions = np.array(designs['conditions'], dtype=int)
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
            # Construct the proper filter
            fc2 = 1 / isi_maxs[k]
            w2 = 2 * fc2 / fs
            if w2 > 1:
                warnings.warn("w2 was automatically fixed to 1.")
                w2 = 1
            b, a = signal.iirfilter(nf, [w1, w2], rs=80, rp=0, btype='band', analog=False, ftype='butter')

            # compute efficiency of all examples
            if np.mod(k, nbr_tests / 10) == 0:
                print("%ds: contrast %d/%d - efficiency evaluation %d/%d" %
                      (time.time() - t, i + 1, nbr_contrasts, k + 1, nbr_tests))

            X = design_matrix(tr, conditions[k], onsets[k], durations, final_soa=isi_maxs[k])

            # Filter each columns to compute efficiency only on the bold signal bandwidth
            for j in range(X.shape[1] - 1):
                X[j] = signal.filtfilt(b, a, X[j])

            # Compute efficiency of this design for this contrast
            efficiencies[i, k] = efficiency(X, c)

    np.save(op.join(output_file), efficiencies)
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
        generate_designs(sys.argv[2])

    elif function == 'compute_efficiencies':
        """ example : python /hpc/banco/bastien.c/python/designer/design_efficiency.py compute_efficiencies
        /hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/params.p
        /hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/designs.npy
        /hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/efficiencies.npy

        frioul_batch '/hpc/crise/anaconda3/bin/python3 /hpc/banco/bastien.c/python/designer/design_efficiency.py
        compute_efficiencies /hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/params.p
        /hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/designs.npy
        /hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/efficiencies.npy' --nodes 20 -c 12

        """
        compute_efficiencies(sys.argv[2], sys.argv[3], sys.argv[4])

    else:
        warnings.warn("Unrecognized function '{}'".format(function))
