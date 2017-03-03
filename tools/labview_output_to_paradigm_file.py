import pandas as pd
import numpy as np
import sys
import warnings
import scipy.io as io
import os.path as op


def set_real_onsets(paradigm, stim_db, is_a_question_step, start_with_iti=True):
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


def labview_to_paradigm(labview_file, stim_db_file, is_question=False):
    data = pd.read_csv(labview_file, sep="\t")

    # Get useful data from the labview output file
    paradigm = {'trial_type': data['CONDITION'], 'file': data['SON'], 'onset': data['ONSETS_MS']/1000.0,
                'expected_answer': data['RESPONSE'], 'answer': data['REPONSE_1'],
                'reaction_time': data['TEMPS_REACTION']/1000.0}

    # Load file info
    print("Reading stim database info file: {}".format(stim_db_file))
    stim_db = pd.read_csv(stim_db_file, sep="\t")

    # Set real onset (based on the real duration of the stimuli)
    good_paradigm = set_real_onsets(paradigm, stim_db, is_a_question_step=is_question, start_with_iti=True)

    # Compute duration to add them to the paradigm after having remove the last onset
    durations = np.array(good_paradigm['onset'][1:]) - np.array(good_paradigm['onset'][:-1])

    # Remove the endding onset (this onset is only used to handle the duration of the last ITI)
    last_paradigm = {}
    for key in paradigm.keys():
        last_paradigm[key] = np.array(good_paradigm[key])[:-1]

    # Set duration of each onset
    last_paradigm['duration'] = durations

    return pd.DataFrame(last_paradigm)


def remove_conditions(para_df, conditions_list):
    tmp_para = para_df.copy()
    for cond in conditions_list:
        idx_to_remove = np.argwhere(np.array(tmp_para['trial_type']) == cond).flatten()
        tmp_para = tmp_para.drop(idx_to_remove)
    return tmp_para


def paradigm_to_csv(para_df, csv_file):
    # CSV file
    para_df.to_csv(csv_file, sep=",", index=False, columns=['onset', 'trial_type', 'file', 'duration',
                                                            'expected_answer', 'answer', 'reaction_time'])
    print("CSV file saved at: {}".format(csv_file))


def paradigm_to_mat(paradigm, mat_file):
    # MATLAB file
    mat_struct = {}
    cond_names = np.unique(paradigm['trial_type'])
    mat_struct['names'] = cond_names
    mat_struct['onsets'] = []
    mat_struct['durations'] = []
    for cond in cond_names:
        mat_struct['onsets'].append(np.array(paradigm['onset'][np.array(paradigm['trial_type']) == cond]).T)

        # Questions are just an event, so duration is set to 0
        if cond == "Question" or cond == "question":
            mat_struct['durations'].append(0)
        else:
            mat_struct['durations'].append(np.array(paradigm['duration'][np.array(paradigm['trial_type']) == cond]).T)

    io.savemat(mat_file, mat_struct)
    print("MATLAB file saved at: {}".format(mat_file))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Labview file to CSV\n")
        print("Create .csv file from the labview output file given by scan center.")
        print("\nOnsets are corrected to match the real duration of each stimulation and last onset is removed. New "
              "file is saved in the same directory than the input file with suffixe.")
        print("\nArgs:\n\t[1]  Labview file\n\t[2]  Database file")
        print("\t[3]  (opt.) If the paradigm use questions set this argument to 1. Default value is 0.\n")
        exit(0)

    lab_file = sys.argv[1]
    db_file = sys.argv[2]

    if len(sys.argv) > 3:
        is_question = sys.argv[3]
    else:
        is_question = False

    paradigm_df = labview_to_paradigm(lab_file, db_file, is_question=is_question)

    # The paradigm structure is now completed and right. Next step is to save it in .csv
    txt_file = lab_file[:-4] + "_original.csv"
    paradigm_to_csv(paradigm_df, txt_file)

    # Filter to remove ISI onsets
    paradigm_df = remove_conditions(paradigm_df, ["ISI", "ITI"])

    filt_txt_file = lab_file[:-4]

    # Save the filtered csv and matlab files
    paradigm_to_csv(paradigm_df, filt_txt_file + "_glm.csv")
    paradigm_to_mat(paradigm_df, filt_txt_file + "_glm.mat")

    # MVPA use one regressor for each file
    paradigm_df['trial_type'] = paradigm_df['file']
    # TODO: order conditions (anne betty chloe)
    paradigm_to_csv(paradigm_df, filt_txt_file + "_mvpa.csv")
    # paradigm_to_mat(paradigm_df, filt_txt_file + "_mvpa.mat")
