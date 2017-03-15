import numpy as np
import warnings


def parse_contrasts_file(filename):
    """Create a dictionnary containing contrasts names and added and subtracted regressors names.
    If the text file, one contrasts by line must be given as follow:
    contrast1_name = regressor1+regressor2+regressor3-regressor4-regressor5-regressor6
    contrast2_name = regressor1
    contrast3_name = regressor4-regressor2

    :param filename: File that contains contrasts descriptions
    :return: Keys name that will be added or substracted of each contrasts name
    """
    file = open(filename, "r")

    contrasts_list = []
    for i, line in enumerate(file):
        # Remove '\n'
        line = line[:-1]

        line = line.split(" = ")
        if len(line) != 2:
            warnings.warn("Contrasts list text file synthax error.")

        contrast_name = line[0]

        # Split added keys from subtracted keys
        to_add = line[1].split('+')
        to_sub = to_add[-1].split('-')
        to_add[-1] = to_sub[0]
        to_sub = to_sub[1:]

        contrasts_list.append({'name': contrast_name, 'added': to_add, 'substracted': to_sub})
    return contrasts_list


def set_contrasts_from_file(orig_contrast, contrasts_file):
    """Construct the suited contrast vectors as defined by the contrasts_file

    :param orig_contrast: Contrasts dictionnary that contains all used regressors
    :param contrasts_file: File that describe which regressors are used for each contrasts.
    :return:
    """

    # Read contrasts list defined in the text file
    contrasts_list = parse_contrasts_file(contrasts_file)

    new_contrasts = {}
    for contrast in contrasts_list:
        to_add = contrast['added']
        to_sub = contrast['substracted']

        contrast_val = orig_contrast[to_add[0]].copy()
        for key_to_add in to_add[1:]:
            contrast_val += orig_contrast[key_to_add]

        for key_to_sub in to_sub:
            contrast_val -= orig_contrast[key_to_sub]

        new_contrasts[contrast['name']] = contrast_val

    return new_contrasts
