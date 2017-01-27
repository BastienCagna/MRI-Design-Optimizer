import pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path as op


def find_best_design(efficiencies, contrasts, n=1):
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

    return i_best


def find_middle_design(efficiencies, contrasts, n=1):
    nbr_tests = efficiencies.shape[1]
    nbr_contrasts = contrasts.shape[0]

    # Distribution for each contrast
    # TODO: Optimize this step be removing for loops if it is possible
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
            print("Low threshold: {}\nHight threshold: {}".format(th_l, th_h))
            # take the last to maximise the efficiency of the first contrast
            i_best = np.argwhere(intersection)[-n:].flatten()
            break

    return i_best


def find_worst_design(efficiencies, contrasts):
    nbr_tests = efficiencies.shape[1]
    nbr_contrasts = contrasts.shape[0]

    # Distribution for each contrast
    eff_rep = np.zeros(efficiencies.shape)
    for c in range(nbr_contrasts):
        for i in range(eff_rep.shape[1]):
            eff_rep[c, i] = np.sum(efficiencies[c, :] < efficiencies[c, i])
    eff_rep = eff_rep / nbr_tests

    # Worst efficiency
    th_v = np.arange(0.0, 1.0, step=0.01)
    nbr_good_tests = np.zeros(th_v.shape)
    i_worst = -1
    for (i, th) in enumerate(th_v):
        bin_eff_rep = eff_rep <= th
        intersection = np.sum(bin_eff_rep, axis=0) == nbr_contrasts
        nbr_good_tests[i] = np.sum(intersection)
        if nbr_good_tests[i] > 0:
            # take the last to maximise the efficiency of the first contrast
            i_worst = np.argwhere(intersection)[-1][0]
            break
    return i_worst, eff_rep[:, i_worst]


def random_dataset(designs_file, efficiencies_file, folds_size, folds_nbr=-1):
    # Read designs files
    designs = np.load(designs_file)
    efficiencies = np.load(efficiencies_file)

    nbr_designs = designs.shape[1]

    designs_folds = []
    efficiencies_folds = []
    indexes_tab = []
    for i in range(folds_nbr):
        indexes = np.random.permutation(nbr_designs)[:folds_size]
        designs_folds.append(designs[:, indexes])
        efficiencies_folds.append(efficiencies[:, indexes])
        indexes_tab.append(indexes)
    return designs_folds, efficiencies_folds, indexes_tab


def compute_result(params_file, designs_file, efficiencies_file, result_file, nbr_tirage = 100):
    nbr_designs = [10, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    nbr_tirage_total = 100000

    params = pickle.load(open(params_file, 'rb'))
    contrasts = params['contrasts']
    i_contrast = 0

    data = []
    for i, k in enumerate(nbr_designs):
        print("computing for {} designs...".format(k))

        dsgns, eff, orig_indexes = random_dataset(designs_file, efficiencies_file, k, nbr_tirage)

        for i_tirage in range(nbr_tirage):
            i_best, percs_best = find_best_design(eff[i_tirage], contrasts)
            i_worst, percs_worst = find_worst_design(eff[i_tirage], contrasts)
            data.append([k, eff[i_tirage][i_contrast][i_best], percs_best[i_contrast], orig_indexes[i_tirage][i_best],
                         eff[i_tirage][i_contrast][i_worst], percs_worst[i_contrast], orig_indexes[i_tirage][i_worst]])
    data = np.array(data)

    to_save = {'data': data, 'nbr_designs': nbr_designs}
    pickle.dump(to_save, open(result_file, "wb"))


if __name__ == "__main__":
    # FIND BEST AND WORST DESIGNS SEVERAL TIMES (100) FOR DIFFERENT NBR OF DESIGNS AND CONSTRAINTS
    # path = '/hpc/banco/bastien.c/data/optim/VL/banks/no_constraint/'
    # result_path = "/hpc/banco/bastien.c/data/optim/VL/test_k/"
    # compute_result(path + "/params.p", path + "designs.npy", path + "efficiencies.npy",
    #                result_path + "data_complex.p")
    #
    # nbr_designs = 100
    # nbr_tirage_total = 100000
    #
    # params = pickle.load(open(path + "params.p", 'rb'))
    # contrasts = params['contrasts']
    # i_contrast = 0
    #
    # dsgns, eff, orig_indexes = random_dataset(path + "designs.npy", path + "efficiencies.npy", nbr_designs, 1)
    #
    # i_best, i_worst_best = find_best_design(eff[0], contrasts)

    path = "/hpc/banco/bastien.c/data/optim/VC/iti_const/"
    eff = np.load(op.join(path, "efficiencies.npy"))
    params = pickle.load((open(op.join(path, "params.p"), "rb")))
    contrasts = params['contrasts']
    i_bests = find_best_design(eff, contrasts, n=1)
    print(i_bests)
    np.save(op.join(path, "bests"), i_bests)

    i_moys = find_middle_design(eff, contrasts, n=1)
    print(i_moys)
    np.save(op.join(path, "moys"), i_moys)
