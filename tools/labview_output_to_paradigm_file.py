import pandas as pd
import numpy as np
import warnings
import scipy.io as io
import os.path as op
import argparse
from os import listdir


def set_real_onsets(paradigm, stim_db, is_a_question_step, start_with_iti=True):
    """

    :param paradigm:
    :param stim_db:
    :param is_a_question_step:
    :param start_with_iti:
    :return:
    """
    onsets = paradigm['onset']
    files = paradigm['file']

    files_list = np.array(stim_db['File'])
    duration = stim_db['Duration']

    # If there is a question after each stimulation the next stimulation step is he third after the previous one
    # Else (there is only ITI) the is the second one.
    if is_a_question_step:
        step = 3
    else:
        step = 2

    if start_with_iti:
        start = 1
    else:
        start = 0

    print("Computing real onsets")
    real_onsets = onsets.copy()
    for i in np.arange(start=start, stop=len(onsets) - 1, step=step):
        # Find the index of played file in the database file
        try:
            idx_db = np.where(files_list == files[i])[0][0]
        except:
            idx_db = -1
            warnings.warn("Can not found '{}' in the database.".format(files[i]))
            exit(-1)

        # Change the onset of the ISI (or the question) to start exactly a the end of the stimulus
        real_onsets[i + 1] = onsets[i] + duration[idx_db]

    output = paradigm
    output['onset'] = real_onsets
    return output


def labview_to_paradigm(paradigm, stim_db, is_question=False):
    # Set real onset (based on the real duration of the stimuli)
    good_paradigm = set_real_onsets(paradigm, stim_db,
                                    is_a_question_step=is_question,
                                    start_with_iti=True)

    # Compute duration to add them to the paradigm after having remove the last
    # onset
    durations = np.array(good_paradigm['onset'][1:]) \
        - np.array(good_paradigm['onset'][:-1])

    # Remove the endding onset (this onset is only used to handle the duration
    # of the last ITI)
    last_paradigm = {}
    for key in paradigm.keys():
        last_paradigm[key] = np.array(good_paradigm[key])[:-1]

    # Set duration of each onset
    last_paradigm['duration'] = durations

    return last_paradigm


def remove_conditions(para_df, conditions_list):
    tmp_para = para_df.copy()

    # Always remove the last onset
    tmp_para = tmp_para.drop(tmp_para.index[len(tmp_para)-1])

    # Remove listed trial_type
    for cond in conditions_list:
        trials = np.array(tmp_para['trial_type'])
        idx_to_remove = np.argwhere(trials == cond).flatten()
        keys = tmp_para['trial_type'].keys()
        tmp_para = tmp_para.drop(keys[idx_to_remove])
    return tmp_para


def paradigm_to_tsv(para_df, tsv_file):
    # CSV file
    para_df.to_csv(tsv_file, sep="\t", index=False,
                   columns=['onset', 'trial_type', 'file', 'duration',
                            'expected_answer', 'answer', 'reaction_time'])
    print("TSV file saved: {}".format(tsv_file))


def paradigm_to_mat(paradigm, mat_file):
    # MATLAB file
    mat_struct = {}

    cond_names = np.unique(paradigm['trial_type'])
    nbr_cond = len(cond_names)
    names_tab = np.empty((nbr_cond,), dtype=object)
    onsets_tab = np.empty((nbr_cond,), dtype=object)
    dur_tab = np.empty((nbr_cond,), dtype=object)
    for i, cond in enumerate(cond_names):
        names_tab[i] = cond

        sel = np.array(paradigm['trial_type']) == cond
        onsets_tab[i] = np.array(paradigm['onset'][sel])

        # Questions are just an event, so duration is set to 0
        if cond == "Question" or cond == "question":
            dur_tab[i] = 0
        else:
            dur_tab[i] = np.array(paradigm['duration'][sel])

    mat_struct['names'] = names_tab
    mat_struct['onsets'] = onsets_tab
    mat_struct['durations'] = dur_tab

    io.savemat(mat_file, mat_struct)
    print("MATLAB file saved: {}".format(mat_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify paradigm files contained in a directory to process "
                    "GLM or MVPA glm.")
    parser.add_argument("file", metavar='FILE', type=str, help="Original file")
    parser.add_argument("-db", dest="db_file", metavar='DATABASE_FILE',
                        type=str, help="Stimuli database file")
    parser.add_argument("-outTemplate", dest="out", type=str, default=None,
                        help="Output filename template")
    parser.add_argument("-ftype", dest="ftype", type=str,
                        default="labview", choices=['tsv', 'labview'],
                        help="Type of the input files")
    parser.add_argument("-noORIG", dest="noORIG", action='store_const',
                        const=True, default=False,
                        help="Unable creation of the original labview file as "
                             ".tsv")
    parser.add_argument("-noGLM", dest="noGLM", action='store_const',
                        const=True, default=False,
                        help="Unable creation of GLM versions")
    parser.add_argument("-noMVPA", dest="noMVPA", action='store_const',
                        const=True, default=False,
                        help="Unable creation of MVPA versions")
    parser.add_argument("-noTSV", dest="noTSV", action='store_const',
                        const=True, default=False,
                        help="Unable creation of .tsv version")
    parser.add_argument("-noMAT", dest="noMAT", action='store_const',
                        const=True, default=False,
                        help="Unable creation of .mat version")
    parser.add_argument("-rmtrials", dest="rmtrials", nargs='*', type=str,
                        default=["ISI", "ITI", "[END]", "END"],
                        help="List of removed trial_types")
    parser.add_argument("-quest", dest="is_question", action='store_const',
                        const=True, default=False,
                        help="Set this flag if there is a question event.")
    args = parser.parse_args()

    print("File type: {}".format(args.ftype))
    print("Removed trials: {}".format(args.rmtrials))

    if args.db_file is not None:
        # Load file info
        stim_db = pd.read_csv(args.db_file, sep="\t")

    indir = op.dirname(args.file)
    files = listdir(indir)
    file = op.basename(args.file)
    print("Searching for: {}\n\tin: {}".format(file, indir))
    tsv_files = []
    for f in files:
        if file in f and ((args.ftype == "labview" and f[-3:] == "txt") or
                          (args.ftype == "tsv" and f[-3:] == "tsv")):
            tsv_files.append(op.join(indir, f))
    if len(tsv_files) == 0:
        print("No file found.")
        exit(0)

    for tsv_file in tsv_files:
        print("\nFile: {}".format(tsv_file))
        data = pd.read_csv(op.join(indir, tsv_file), sep="\t")

        if args.ftype == "labview":
            # Get useful data from the labview output file
            paradigm = {
                'trial_type': data['CONDITION'],
                'file': data['SON'],
                'onset': data['ONSETS_MS'] / 1000.0,
                'expected_answer': data['RESPONSE'],
                'answer': data['REPONSE_1'],
                'reaction_time': data['TEMPS_REACTION'] / 1000.0
            }

            if args.db_file is not None:
                # Get true durations
                paradigm = labview_to_paradigm(paradigm, stim_db,
                                               is_question=args.is_question)
            paradigm_df = pd.DataFrame(paradigm)
        else:
            paradigm_df = pd.DataFrame(data)

        # New files will have same name that input with different suffixes
        new_filename = args.out

        if args.ftype == "labview" and args.noORIG is False:
            # The paradigm structure is now completed and right.
            # Next step is to save it in .tsv
            txt_file = new_filename + "_model-raw_events.tsv"
            paradigm_to_tsv(paradigm_df, txt_file)

        # Filter to remove some unused onsets
        paradigm_df = remove_conditions(paradigm_df, args.rmtrials)

        # Save the filtered csv and matlab files
        if args.noGLM is False:
            if args.noTSV is False:
                paradigm_to_tsv(paradigm_df, new_filename +
                                "_model-glm_events.tsv")
            if args.noMAT is False:
                paradigm_to_mat(paradigm_df, new_filename +
                                "_model-glm_events.mat")

        # MVPA use one regressor for each file
        if args.noMVPA is False and args.ftype == "labview":
            paradigm_df['trial_type'] = paradigm_df['file']
            # TODO: order conditions (anne betty chloe)
            if args.noTSV is False:
                paradigm_to_tsv(paradigm_df, new_filename +
                                "_model-mvpa_events.tsv")
            if args.noMAT is False:
                paradigm_to_mat(paradigm_df, new_filename +
                                "_model-mvpa_events.mat")
