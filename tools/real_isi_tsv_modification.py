#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import warnings


def get_real_isi_onsets(run_file, stim_db_file, output_file, is_a_question_step=False, start_with_ITI=True):
    """

    :param run_file:
    :param stim_db_file:
    :param output_file:
    :return:
    """
    print("Reading labview file: {}".format(run_file))
    run = pd.read_csv(run_file, sep="\t")

    print("Reading stim database info file: {}".format(stim_db_file))
    stim_db = pd.read_csv(stim_db_file, sep="\t")

    onsets = run['ONSETS_MS']
    files = run['SON']

    files_list = np.array(stim_db['File'])
    duration = stim_db['Duration']

    if is_a_question_step:
        step = 3
    else:
        step = 2

    if start_with_ITI:
        start = 1
    else:
        start = 0

    print("Computing real onsets")
    real_onsets = onsets.copy()
    for i in np.arange(start=start, stop=len(onsets)-1, step=step):
        # Find the index of played file in the database file
        try:
            idx_db = np.where(files_list == files[i])[0][0]
        except:
            warnings.warn("Can not found '{}' in the database.".format(files[i]))
            exit(-1)

        # Change the onset of the ISI (or the question) to start exactly a the end of the stimulus
        real_onsets[i+1] = onsets[i] + np.round(duration[idx_db] * 1000)

    output = run
    output['ONSETS_MS'] = real_onsets

    output.to_csv(output_file, sep="\t", index=False)
    print("New file saved at: {}".format(output_file))


if __name__ == "__main__":
    r_file = sys.argv[1]
    db_file = sys.argv[2]
    out_file = sys.argv[3]
    if len(sys.argv) > 4:
        is_question = bool(sys.argv[4])
    else:
        is_question = False

    get_real_isi_onsets(r_file, db_file, out_file, is_a_question_step=is_question, start_with_ITI=True)
