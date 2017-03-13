#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys


def to_labview(output_filename, design, trig_by_sec=13, last_iti_sec=4.0, final_word="END",
               question_cond="", question_txt="", question_dur=0, responses_idx=None, iti_text="+"):
    """

    :param output_filename: Ouput .csv filename.
    :param design: Design.
    :param trig_by_sec: Number of triggers by seconds.
    :param last_iti_sec: Last ITI (in second).
    :param final_word: Final printed text.
    :param question_cond: Question condition name.
    :param question_txt: Question text.
    :param question_dur: Question duration (in second). This timing is substracted to ITI.
    :param responses_idx: Array of attempted response. Each response is represented as negative integer.
    :param iti_text: Text printed for each ITI.
    :return:
    """
    # Avoid blank space
    iti_text = iti_text.strip()

    onsets = design['onset']
    groups = design['trial_group']
    files = design['files']
    durations = design['duration']
    iti_vect = design['ITI']

    question_tr = int(question_dur * trig_by_sec)

    # CSV columns
    exp_conds = []
    exp_filenames = []
    exp_durs = []
    exp_text = []
    exp_response = []

    # Intial ITI
    iti_tr = int(onsets[0] * trig_by_sec)
    exp_conds.append("ITI")
    exp_text.append(iti_text)
    exp_filenames.append("[Nothing]")
    exp_durs.append(iti_tr)
    exp_response.append(0)

    for i in range(len(onsets)):
        # Onset
        stim_dur_tr = int(np.ceil(durations[i] * trig_by_sec))
        exp_conds.append(groups[i])
        exp_text.append("")
        exp_filenames.append(files[i].strip())
        exp_durs.append(stim_dur_tr)
        exp_response.append(0)

        # Optional fixed question time
        if question_dur > 0:
            exp_conds.append(question_txt)
            exp_text.append(question_cond)
            exp_filenames.append("[Question]")
            exp_durs.append(question_tr)
            exp_response.append(responses_idx[i])

        # ITI
        iti_tr = int(iti_vect[i] * trig_by_sec)
        exp_conds.append("ITI")
        exp_text.append(iti_text)
        exp_filenames.append("[Nothing]")
        exp_durs.append(iti_tr)
        exp_response.append(0)

    # Final Word
    iti_tr = int(last_iti_sec * trig_by_sec)
    exp_conds.append("[Final step]")
    exp_text.append(final_word.strip())
    exp_filenames.append("[Nothing]")
    exp_durs.append(iti_tr)
    exp_response.append(0)

    if question_dur > 0:
        unused = np.zeros((len(onsets)*3 + 2, ), dtype=int)
    else:
        unused = np.zeros((len(onsets)*2 + 2, ), dtype=int)

    df = pd.DataFrame({"CONDITION": exp_conds, "TEXT": exp_text, "BMP_NAME": unused, "SON": exp_filenames,
                       "DURATION_TRIGGERS": exp_durs, "RESPONSE": exp_response, "UNUSED_1": unused, "UNUSED_2": unused})

    df.to_csv(output_filename,
              columns=["CONDITION", "TEXT", "BMP_NAME", "SON", "DURATION_TRIGGERS", "RESPONSE", "UNUSED_1", "UNUSED_2"],
              index=False)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(" Design file to labview file ")
        print("\n *** Args ***")
        print("[1]  Output file name (TSV)")
        print("[2]  Input design file (dictionnary as .pck)")
        print("[3]  Trigger frequency")
        print("[4]  Last ITI duration (in seconds)")
        print("[5]  ITI text")
        print("[6]  End text")
        print("If there is a question after each onset:")
        print("[7]  Question duration (in seconds)")
        print("[8]  Question text")
        print("[9]  Condition name for questions")
        print("[10] Response file (.npy)")
    output_file = sys.argv[1]
    dsgn = pd.read_pickle(sys.argv[2])
    trig_freq = float(sys.argv[3])
    last_iti = float(sys.argv[4])
    iti_txt = sys.argv[5]
    end_wrd = sys.argv[6]
    if len(sys.argv) > 7:
        q_dur = float(sys.argv[7])
        q_txt = sys.argv[8]
        q_cond = sys.argv[9]
        resp = np.load(sys.argv[10])
    else:
        q_dur = 0
        q_txt = ""
        q_cond = ""
        resp = None

    to_labview(output_file, dsgn, trig_by_sec=trig_freq, last_iti_sec=last_iti, final_word=end_wrd,
               question_cond=q_cond, question_txt=q_txt, question_dur=q_dur, responses_idx=resp, iti_text=iti_txt)