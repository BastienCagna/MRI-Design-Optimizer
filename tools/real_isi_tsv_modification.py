#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys


def get_real_isi_onsets(run_file, stim_db_file, output_file, is_a_question_step=False, start_with_ISI=True):
    """

    :param run_file:
    :param stim_db_file:
    :param output_file:
    :return:
    """
    run = pd.read_csv(run_file, sep="\t")
    stim_db = pd.read_csv(stim_db_file, sep="\t")

    onsets = run['ONSETS_MS']
    files = run['SON']

    files_list = np.array(stim_db['File'])
    duration = stim_db['Duration']

    if is_a_question_step:
        step = 3
    else:
        step = 2

    if start_with_ISI:
        start = 1
    else:
        start = 0

    real_onsets = onsets.copy()
    for i in np.arange(start=start, stop=len(onsets)-1, step=step):
        # Find the index of played stimulus in the database file
        idx_db = np.where(files_list == files[i])[0][0]

        # Change the onset of the ISI to start exactly a the end of the stimulus
        real_onsets[i+1] = onsets[i] + int(duration[idx_db] * 1000)

    output = run
    output['ONSETS_MS'] = real_onsets

    output.to_csv(output_file, sep="\t")


if __name__ == "__main__":
    r_file = sys.argv[1]
    db_file = sys.argv[2]
    out_file = sys.argv[3]
    if len(sys.argv) > 4:
        is_question = bool(sys.argv[4])

    get_real_isi_onsets(r_file, db_file, out_file, is_a_question_step=is_question, start_with_ISI=True)
