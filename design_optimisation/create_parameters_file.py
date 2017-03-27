#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

    Create the parameters file needed for the design optimizer pipeline.

"""
import numpy as np
import pickle
import os.path as op

from design_optimisation.contrast_definition import parse_contrasts_file


def write_parameters_file(conditions_names, cond_of_files, groups, group_names, contrasts_def_file,
                          files_duration, files_list, iti_file, nbr_designs, tmp, tmn, tr, work_dir,
                          responses=None, question_dur=0, verbose=True):
    """
    Create the parameters file containning general design paradigm and other parameters needed by the pipeline.

    Arguments
    =========

    :param conditions_names: Name of each condition.
    :param cond_of_files: Conditions of each file of the files_list.
    :param groups: Vector that give the group of each condition.
    :param group_names: Name of each group.
    :param contrasts_def_file: File that describe contrasts
    :param files_duration: Duration of each file (in seconds).
    :param files_list: List of .wav stimuli files.
    :param iti_file: Path to a numpy file that contains all used ITI in a single vector.
    :param nbr_designs: Number of designs to generate.
    :param tmp: Transition Matrix (previous) as defined by Hanson, 2015 (DOI:
    B978-0-12-397025-1.00321-3) p.489 to p.494.
    :param tmn: Transition Matrix (next) as defined by Hanson, 2015 (DOI:
    B978-0-12-397025-1.00321-3) p.489 to p.494.
    :param tr: Repetition time (in seconds).
    :param work_dir: Working directory. Output file of the pipeline will be saved in this directory.
    :param responses: 1D array containing the expected response of each stimulus.
    :param question_dur: Question duration (in second).
    :param verbose: (optional) Print all parameters. Default: False.

    Output
    ======
    :return: Nothing. The parameters file is saved in the working directory.

    """

    # Ensure that variable are numpy array (to make research easier)
    tmp = np.array(tmp)
    tmn = np.array(tmn)
    files_duration = np.array(files_duration)
    files_list = np.array(files_list)
    cond_of_files = np.array(cond_of_files)
    conditions_names = np.array(conditions_names)
    groups = np.array(groups)
    group_names = np.array(group_names)

    ucond = np.unique(cond_of_files)
    count_by_cond = {}
    nbr_event = 0
    for cond in ucond:
        count_of_cond = np.sum(cond_of_files==cond)
        count_by_cond[cond] = count_of_cond
        nbr_event += count_of_cond

    contrasts_def = parse_contrasts_file(contrasts_def_file)

    # Save parameters in a dictionnary
    params = {
        'cond_names': conditions_names,
        'cond_counts': count_by_cond,
        'cond_of_files': cond_of_files,
        'cond_groups': groups,
        'group_names': group_names,
        'contrasts_def': contrasts_def,
        'files_duration': files_duration,
        'files_list': files_list,
        'ITI_file': iti_file,
        'nbr_designs': nbr_designs,
        'nbr_events': nbr_event,
        'TMp': tmp,
        'TMn': tmn,
        'tr': tr,
        'responses': responses,
        'responses_dur': question_dur,
        'work_dir': work_dir
    }
    if verbose:
        print(params)

    pickle.dump(params, open(op.join(work_dir, "params.pck"), "wb"))
    return
