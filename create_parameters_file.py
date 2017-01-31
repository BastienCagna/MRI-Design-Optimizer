#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

    Use this function to create the parameters file needed for the design optimizer pipeline.

"""
import numpy as np
import scipy.io.wavfile as wf
import pickle
import os.path as op


def write_parameters_file(conditions_names, cond_of_files, groups, contrasts, contrast_names, files_list, files_path,
                          iti_file, nbr_designs, tmp, tmn, tr, output_path, output_file="params.p", verbose=False):
    """
    Create the design configuration file containning parameters that's will be used by the pipeline.

    :param output_file: Working directory.
    :param conditions_names: Name of each condition.
    :param cond_of_files: Conditions of each file of the files_list.
    :param groups: Vector that give the group of each condition.
    :param contrasts: List of contrasts. Each row of the array is a contrast. Last coeffcient of each row corresponds to
                      the intercept.
    :param contrast_names: Name of each contrasts (for final figures).
    :param files_list: List of .wav stimuli files.
    :param files_path: Path to .wav stimuli files.
    :param iti_file: Path to a numpy file that contains all used ITI in a single vector.
    :param nbr_designs: Number of designs to generate.
    :param tmp: Transition Matrix (previous) as defined by Hanson, 2015.
    :param tmn: Transition Matrix (next) as defined by Hanson, 2015.
    :param tr: Repetition tme (in seconds).
    :param output_path: (optional) Parameters file name. Default: params.p
    :param verbose: (optional) Print all parameters. Default: False.
    :return: Nothing. The paramter file is saved in the output_path.
    """

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
        'ITI_file': iti_file,
        'nbr_designs': nbr_designs,
        'TMp': tmp,
        'TMn': tmn,
        'tr': tr,
        'output_path': output_path
    }
    if verbose:
        print(params)

    pickle.dump(params, open(op.join(output_path, output_file), "wb"))
    return

