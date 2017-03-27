#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import sys
import time
import warnings
from scipy import signal
from nistats.design_matrix import make_design_matrix

from design_optimisation.contrast_definition import get_contrasts_list


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

    x = make_design_matrix(frame_times, pd.DataFrame(design), drift_model='blank')

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


def compute_efficiencies(tr, designs, contrasts_def, isi_maxs, nf=6, fc1=1 / 120, verbose=False):
    """Compute efficiencies of each designs and each contrasts.

    Each regressor of the design matrix (each one corresponding to a condition) is filtered by a
    butterwotrh filter and
    then the efficiency of the resulting matrix is computed.

    :param tr:
    :param designs:
    :param contrasts_def:
    :param isi_maxs:
    :param nf: (optional) Order of the butterworth filter. Default: 9.
    :param fc1: (optional) High pass cutting frequency of the filter (in Hertz). Default: 1/120 Hz.
    :param verbose: (optional) Set to True to get more printed details of the running. Default:
    False.
    :return: efficiencies matrix (each row contains efficiencies of one design for all the
    contrasts)
    """
    nbr_designs = len(designs)
    nbr_contrasts = len(contrasts_def.keys())
    eff_arr = np.zeros((nbr_contrasts, nbr_designs))

    t = time.time()

    for k, design in enumerate(designs):
        if verbose and np.mod(k, nbr_designs / 100) == 0:
            print("[{:.3f}s] {}%".format(time.time() - t, int(100 * k / nbr_designs)))

        d_matrix = filtered_design_matrix(tr, design, isi_maxs[k], fc1=fc1, nf=nf,
                                          filt_type='highpass')

        # Determine the place of all condition in the contrast vector
        contrasts = get_contrasts_list(contrasts_def, design_matrix=d_matrix)

        for (i, c_name) in enumerate(contrasts.keys()):
            # Compute efficiency of this design for this contrast
            eff_arr[i, k] = efficiency(d_matrix, contrasts[c_name])

    return eff_arr


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Compute efficiency of each design for each contrasts.")
        exit(0)

    print("Efficiencies computation")

    params = pickle.load(open(sys.argv[2], "rb"))
    data = pickle.load(open(sys.argv[3], "rb"))

    efficiencies = compute_efficiencies(params['tr'], data['designs'], params['contrasts'],
                                        data['isi_maxs'], verbose=True)

    np.save(sys.argv[4], efficiencies)
    print("Efficiencies have been saved to: {}".format(sys.argv[4]))
