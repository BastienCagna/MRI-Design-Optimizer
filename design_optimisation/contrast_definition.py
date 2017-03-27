import numpy as np
import warnings


def parse_contrasts_file(filename):
    """Create a dictionnary containing contrasts names and added and subtracted regressors names.
    If the text file, one contrasts by line must be given as follow:
    contrast1_name = regressor1+regressor2+regressor3-regressor4-regressor5-regressor6
    contrast2_name = regressor1
    contrast3_name = regressor4-regressor2

    :param filename: File that contains contrasts descriptions
    :return: Keys names that will be added or substracted of each contrasts name.
    """
    file = open(filename, "r")

    contrasts_def = {}
    for i, line in enumerate(file):
        if line != '\n':
            # Remove '\n'
            line = line[:-1]

            line = line.split(" = ")
            if len(line) != 2:
                warnings.warn("{}".format(line))
                warnings.warn("Contrasts list text file synthax error.")
                return None

            contrast_name = line[0]

            # Split added keys from subtracted keys
            to_add = line[1].split('+')
            to_sub = to_add[-1].split('-')
            to_add[-1] = to_sub[0]
            to_sub = to_sub[1:]

            contrasts_def[contrast_name] = {'added': to_add, 'substracted': to_sub}
    return contrasts_def


def get_contrasts_list(contrasts_def, regressors_contrasts=None, design_matrix=None):
    """Construct the suited contrast vectors as defined by the contrasts_file

    :param regressors_contrasts: Contrasts dictionnary that contains one contrasts by regressor
    :param contrasts_def: Suited contrasts definitions
    :param design_matrix: If regressors_contrasts are not given, design matrix must be given to
    find the regressors order in the matrix's dataframe.
    :return: Suited contrasts
    """
    if regressors_contrasts is None:
        # TODO : add and error if design matrix is not given
        regressors = design_matrix.keys()
        nbr_regressors = len(regressors)
        regressors_contrasts = {}
        for i_reg, reg_name in enumerate(regressors):
            regressors_contrasts[reg_name] = np.zeros((nbr_regressors,))
            regressors_contrasts[reg_name][i_reg] = 1

    new_contrasts = {}
    for contrast_name in contrasts_def.keys():
        to_add = contrasts_def[contrast_name]['added']
        to_sub = contrasts_def[contrast_name]['substracted']

        contrast_val = regressors_contrasts[to_add[0]].copy()
        for key_to_add in to_add[1:]:
            contrast_val += regressors_contrasts[key_to_add]

        for key_to_sub in to_sub:
            contrast_val -= regressors_contrasts[key_to_sub]

        new_contrasts[contrast_name] = contrast_val

    return new_contrasts
