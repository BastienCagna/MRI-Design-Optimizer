#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import sys
import time


def add_answer_event(paradigm, answer_duration):
    """
        Add response event after each onset of the given design matrix

        :param paradigm: Original paradigm (as dictionary)
        :param answer_duration: Duration of the answer event (in second)

        :return: The modified paradigm
    """

    # Define anwsers onsets
    nbr_event = len(paradigm['onset'])
    ans_onsets = paradigm['onset'] + paradigm['duration']
    ans_duration = np.ones((nbr_event, )) * answer_duration
    ans_iti = paradigm['ITI'] - answer_duration
    ans_files = np.repeat([""], nbr_event)
    ans_trial_type = np.repeat(["answer"], nbr_event)
    ans_trial_idx = -1 * np.ones((nbr_event,))
    ans_trial_grp = np.repeat(["answer"], nbr_event)

    # New paradigm
    modified_pgm = paradigm.copy()
    modified_pgm['onset'] = np.concatenate((modified_pgm['onset'], ans_onsets))
    modified_pgm['ITI'] = np.zeros((nbr_event, ))
    modified_pgm['ITI'] = np.concatenate((modified_pgm['ITI'], ans_iti))
    modified_pgm['duration'] = np.concatenate((modified_pgm['duration'], ans_duration))
    modified_pgm['files'] = np.concatenate((modified_pgm['files'], ans_files))
    modified_pgm['trial_type'] = np.concatenate((modified_pgm['trial_type'], ans_trial_type))
    modified_pgm['type_idx'] = np.concatenate((modified_pgm['type_idx'], ans_trial_idx))
    modified_pgm['trial_group'] = np.concatenate((modified_pgm['trial_group'], ans_trial_grp))

    # Isi max update
    #isi

    return modified_pgm


if __name__ == "__main__":
    file = sys.argv[1]
    ans_dur = float(sys.argv[2])
    output_file = sys.argv[3]

    data = pickle.load(open(file, 'rb'))
    designs = data['designs']
    isi_maxs = data['isi_maxs']
    nbr_designs = len(designs)

    print("{} designs to process".format(nbr_designs))

    t_start = time.time()
    new_paradigm = []
    for i, design in enumerate(designs):
        if np.mod(i, nbr_designs/100) == 0:
            print("[{:.02f}s] {}%".format(time.time() - t_start, int(100*i/nbr_designs)))
        new_paradigm.append(add_answer_event(design, ans_dur))

    print("Writting {}".format(output_file))
    data['designs'] = new_paradigm
    # TODO: update isi_maxs
    data['isi_maxs'] = isi_maxs
    pickle.dump(data, open(output_file, "wb"))
