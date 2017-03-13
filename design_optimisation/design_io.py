import numpy as np
import pickle
import sys
import pandas as pd


def dict_to_vect(dic):
    """Convert a dictionnary in wich keys doesn't matter too an array.

    :param dic: The dictionnary.
    :return: A numpy array.
    """
    vect = []
    for item in dic.keys():
        vect.append(dic[item])
    return np.array(vect)


def design_dataframe2dic(design_df):
    """Convert a design from pandas dataframe to dictionnary.

    :param design_df: Design as pandas' dataframe
    :return: Design as dictionary
    """
    onset_v = dict_to_vect(design_df['onset'])
    dur_v = dict_to_vect(design_df['duration'])
    iti_v = dict_to_vect(design_df['ITI'])
    trial_type = dict_to_vect(design_df['trial_type'])
    type_idx = dict_to_vect(design_df['type_idx'])
    trial_grp = dict_to_vect(design_df['trial_group'])
    files = dict_to_vect(design_df['files'])
    design = {'onset': onset_v, 'trial_type': trial_type, 'duration': dur_v, 'ITI': iti_v, 'type_idx': type_idx,
              'trial_group': trial_grp, 'files': files}
    return design


def extract_design_from_list(designs_file, design_index, design_file):
    """Save a design of the list in a new pickle file.

    :param designs_file: List of design as pickle file. (List of dictionaries)
    :param design_index: Index of the design to extract.
    :param design_file: New design file (.pck)
    :return: Nothing.
    """
    designs = pickle.load(open(designs_file, "rb"))
    design = designs['designs'][design_index]
    pickle.dump(design, open(design_file, "wb"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nDesign IO\n\n[1] Function name\n\n'extract' function:\n\t[2]  Listed designs file (.pck)\n\t[3]  "
              "Index of the design to extract\n\t[4]  Design file name (.pck)")
        exit(0)

    function = sys.argv[1]

    if function == "extract":
        extract_design_from_list(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        print("Unrecognized function.")
