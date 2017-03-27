#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io.wavfile as wf
import os.path as op
import numpy as np
import pandas as pd
import sys


def wave_files_durations(files_list, files_path=None):
    """Compute wav signal duration for each files.

    :param files_list: Wave files list (filename doesn't contains .wav extension).
    :param files_path: Path of the directory that contains waves files. The path is the same for every files.
    :return: The list of durations (in seconds) corresponding to the input file list.
    """
    durations = []
    # For each file of the list
    for file in files_list:
        # If a path is defined, read at [path]/[file] else read at [file]
        if files_path is not None:
            fs, x = wf.read(op.join(files_path, file + '.wav'))
        else:
            fs, x = wf.read(file)
        # Duration is [nbr of samples] / [samplerate]
        durations.append(len(x)/float(fs))

    return np.array(durations)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\nAdd duration column to the stimuli database file of a BIDS database.\n")

    stim_db_file = sys.argv[1]
    stim_dir = sys.argv[2]
    if len(sys.argv) > 3:
        separateur = sys.argv[3]
    else:
        separateur = ','

    print("csv file: {}".format(stim_db_file))
    stim_db_file = pd.read_csv(open(stim_db_file, 'r'), sep=separateur)

    # Get the simuli file list from the stimuli database description file
    files = stim_db_file['file']

    # Compure duration of each file
    stim_db_file['duration'] = wave_files_durations(files,files_path=stim_dir)

    # Add durations to the stimuli database description file
    stim_db_file.to_csv(open(stim_db_file, 'w'), index=False, sep=separateur)

