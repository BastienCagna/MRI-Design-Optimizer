#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os.path as op


def to_labview(output_filename, design, trig_by_sec=13, last_isi_sec=4.0, final_word ="END",
               question_cond="", question_txt="", question_dur=0, responses_idx=None, isi_text="+"):
    """

    :param output_filename: Ouput .csv filename.
    :param design: Design.
    :param trig_by_sec: Number of triggers by seconds.
    :param last_isi_sec: Last ISI (in second).
    :param final_word: Final printed text.
    :param question_cond: Question condition name.
    :param question_txt: Question text.
    :param question_dur: Question duration (in second). This timing is substracted to ISI.
    :param responses_idx: Array of attempted response. Each response is represented as negative integer.
    :param isi_text: Text printed for each ISI.
    :return:
    """

    onsets = design['onset']
    groups = design['trial_group']
    files = design['files']
    durations = design['duration']

    isi_vect = onsets[1:] - onsets[:-1] - question_dur

    question_tr = int(question_dur * trig_by_sec)

    # CSV columns
    exp_conds = []
    exp_filenames = []
    exp_durs = []
    exp_text = []
    exp_response = []

    # Intial ISI
    isi_tr = int(onsets[0] * trig_by_sec)
    exp_conds.append("ISI")
    exp_text.append(isi_text)
    exp_filenames.append("[Nothing]")
    exp_durs.append(isi_tr)
    exp_response.append(0)

    for i in range(len(onsets)):
        # Onset
        stim_dur_tr = int(np.ceil(durations[i] * trig_by_sec))
        exp_conds.append(groups[i])
        exp_text.append("")
        exp_filenames.append(files[i])
        exp_durs.append(stim_dur_tr)
        exp_response.append(0)

        # Optional fixed question time
        if question_dur > 0:
            exp_conds.append(question_txt)
            exp_text.append(question_cond)
            exp_filenames.append("[Question]")
            exp_durs.append(question_tr)
            exp_response.append(responses_idx[i])

        # ISI
        isi_tr = int(isi_vect[i] * trig_by_sec)
        exp_conds.append("ISI")
        exp_text.append(isi_text)
        exp_filenames.append("[Nothing]")
        exp_durs.append(isi_tr)
        exp_response.append(0)

    # Final Word
    isi_tr = int(last_isi_sec * trig_by_sec)
    exp_conds.append("[Final step]")
    exp_text.append(final_word)
    exp_filenames.append("[Nothing]")
    exp_durs.append(isi_tr)
    exp_response.append(0)

    unused = np.zeros((len(onsets)*3 + 2, ), dtype=int)
    df = pd.DataFrame({"CONDITION": exp_conds, "TEXT": exp_text, "BMP_NAME": unused, "SON": exp_filenames,
                       "DURATION_TRIGGERS": exp_durs, "RESPONSE": exp_response, "UNUSED_1": unused, "UNUSED_2": unused})

    df.to_csv(output_filename,
              columns=["CONDITION", "TEXT", "BMP_NAME", "SON", "DURATION_TRIGGERS", "RESPONSE", "UNUSED_1", "UNUSED_2"],
              index=False)


# designs_indexes = [86180, 45929, 56492, 32562]
# for d in designs_indexes:
#     designs = np.load("/hpc/banco/bastien.c/data/optim/identification/final/designs.npy")
#     design = designs[:, d]
#     onsets = design[0]
#     conds = design[1]
#
#     output_file = "/hpc/banco/bastien.c/data/optim/identification/final/design_{:06d}_export.csv".format(d)