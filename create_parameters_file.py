#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

    Use this function to create the parameters file needed for the design optimizer pipeline.

"""
import numpy as np
import scipy.io.wavfile as wf
import pickle
import os.path as op


# TODO: add pages number of the Hanson article in comments
def write_parameters_file(conditions_names, cond_of_files, groups, group_names, contrasts, contrast_names,
                          files_duration, files_list, iti_file, nbr_designs, tmp, tmn, tr, work_dir,
                          output_file="params.pck", responses=None, responses_dur=0, verbose=True):
    """
    Create the design configuration file containning parameters that's will be used by the pipeline.

    :param output_file: Working directory.
    :param conditions_names: Name of each condition.
    :param cond_of_files: Conditions of each file of the files_list.
    :param groups: Vector that give the group of each condition.
    :param contrasts: List of contrasts. Each row of the array is a contrast. Last coeffcient of each row corresponds to
                      the intercept.
    :param contrast_names: Name of each contrasts (for final figures).
    :param files_duration: Duration of each file (in seconds).
    :param files_list: List of .wav stimuli files.
    :param iti_file: Path to a numpy file that contains all used ITI in a single vector.
    :param nbr_designs: Number of designs to generate.
    :param tmp: Transition Matrix (previous) as defined by Hanson, 2015 (DOI: B978-0-12-397025-1.00321-3).
    :param tmn: Transition Matrix (next) as defined by Hanson, 2015 (DOI: B978-0-12-397025-1.00321-3).
    :param tr: Repetition time (in seconds).
    :param verbose: (optional) Print all parameters. Default: False.
    :return: Nothing. The parameters file is saved in the output_path.
    """
    tmp = np.array(tmp)
    tmn = np.array(tmn)
    contrasts = np.array(contrasts)

    # Count for each condition, how many times she's appearing
    cond_of_files = np.array(cond_of_files)
    ucond = np.unique(cond_of_files)
    nbr_cond = len(ucond)
    count_by_cond = np.zeros((nbr_cond,))
    for cond in ucond:
        count_by_cond[cond] = np.sum(cond_of_files==cond)

    # Save conditions and onsets
    params = {
        'cond_names': conditions_names,
        'cond_counts': count_by_cond,
        'cond_of_files': cond_of_files,
        'cond_groups': groups,
        'group_names': group_names,
        'contrasts': contrasts,
        'contrasts_names': contrast_names,
        'files_duration': files_duration,
        'files_list': files_list,
        'ITI_file': iti_file,
        'nbr_designs': nbr_designs,
        'TMp': tmp,
        'TMn': tmn,
        'tr': tr,
        'responses': responses,
        'responses_dur': responses_dur,
        'work_dir': work_dir
    }
    if verbose:
        print(params)

    pickle.dump(params, open(op.join(work_dir, output_file), "wb"))
    return

