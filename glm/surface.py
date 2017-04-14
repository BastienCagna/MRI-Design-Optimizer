"""
First level analysis of localizer dataset
=========================================
Full step-by-step example of fitting a GLM to experimental data and visualizing
the results.
More specifically:
1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. a mask of the useful brain volume is computed
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation)
"""
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os.path as op

import sys
from nilearn import plotting
import nibabel.gifti as ng

sys.path[3] = "/hpc/banco/bastien.c/python/nistats/"
print("\n/!\ Warning: system path have been changed /!\\ \n")

from nistats.first_level_model import FirstLevelSurfaceModel
from nistats.design_matrix import make_design_matrix, plot_design_matrix

from design_optimisation.contrast_definition import get_contrasts_list, parse_contrasts_file

# def run_glm(tr, slice_time_ref, paradigm_file, motion_file, fmri_img):
#     # Read the paradigm file
#     paradigm = pd.read_csv(paradigm_file, sep=',', index_col=None)
#
#     # Read the motion file
#     motion_cols = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
#     motion = pd.read_csv(motion_file, sep="  ", index_col=None, names=motion_cols, engine='python')
#     nbr_vol = len(motion[motion.keys()[0]])
#     print("Number of scans in motion file: {}".format(nbr_vol))
#
#     # **** Build the design matrix ***
#     # Frame timing is determined with number of scans registered in the motion file
#     t_vect = np.arange(nbr_vol) * tr
#     # Create the design matrix without motion drift regressors
#     d_matrix = make_design_matrix(t_vect, paradigm=paradigm, drift_model='blank')
#     # Add motion regressors
#     for col in motion_cols:
#         for i, t in enumerate(t_vect):
#             motion[col][t] = motion[col][i]
#         d_matrix[col] = motion[col]
#
#     # Plot the design matrix
#     plot_design_matrix(d_matrix)
#
#     # **** Perform first level glm ****
#     # Setup and fit GLM
#     first_level_surf_model = FirstLevelSurfaceModel(tr, slice_time_ref, hrf_model='glover + derivative')
#     first_level_surf_model = first_level_surf_model.fit(fmri_img, design_matrices=d_matrix)
#
#     # **** Estimate contrasts ****
#     # Specify the contrasts
#     design_matrix = first_level_surf_model.design_matrices_[0]
#     contrast_matrix = np.eye(design_matrix.shape[1])
#     contrasts = dict([(column, contrast_matrix[i])
#                       for i, column in enumerate(design_matrix.columns)])
#
#     return first_level_surf_model, contrasts


def run_glm(tr, slice_time_ref, paradigm_file, fmri_img, motion_file=None):
    """Run a GLM on surface usingon the given surface images.
    1)    Create a design matrix using the paradigm file
    1bis) Add the motion regressors if the motionn file is given
    2)    Create a first_level_surf_model object and fit the GLM
    3)    Get all possible individual effect contrasts
    :param tr: Repetition time
    :param slice_time_ref: Number of slice by TR
    :param paradigm_file: CSV file that list onsets and at least associated duration and condition
    :param fmri_img: Name of the gifti file that contains surface fMRI data
    :param motion_file: Text file that contains (6) estimated motion regressors
    :return: first_level_surf_model object + original contrasts ([[1 0 0 ... 0], [0 1 0 ... 0]...])
    """
    # Read the paradigm file
    paradigm = pd.read_csv(paradigm_file, sep=',', index_col=None)

    if motion_file is not None:
        # Create a design matrix and add previously estimated motion regressors
        # Read the motion file
        motion_cols = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
        motion = pd.read_csv(motion_file, sep="  ", index_col=None, names=motion_cols, engine='python')
        nbr_vol = len(motion[motion.keys()[0]])
        print("Number of scans in motion file: {}".format(nbr_vol))

        # **** Build the design matrix ***
        # Frame timing is determined with number of scans registered in the motion file
        t_vect = np.arange(nbr_vol) * tr
        # Create the design matrix without motion drift regressors
        d_matrix = make_design_matrix(t_vect, paradigm=paradigm, drift_model='blank')
        # Add motion regressors
        for col in motion_cols:
            for i, t in enumerate(t_vect):
                motion[col][t] = motion[col][i]
            d_matrix[col] = motion[col]
    else:
        # Find the number of voulume in the .gii file
        gii_imgs = ng.read(fmri_img)
        nbr_vol = len(gii_imgs.darrays)
        del gii_imgs
        print("Number of scans : {}".format(nbr_vol))
        t_vect = tr * np.arange(nbr_vol)
        d_matrix = make_design_matrix(t_vect, paradigm=paradigm, drift_model='blank')

    # **** Perform first level glm ****
    # Setup and fit GLM
    first_level_surf_model = FirstLevelSurfaceModel(tr, slice_time_ref, hrf_model='glover + derivative')
    first_level_surf_model = first_level_surf_model.fit(fmri_img, design_matrices=d_matrix)

    # **** Estimate contrasts ****
    # Specify the contrasts
    design_matrix = first_level_surf_model.design_matrices_[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])

    return first_level_surf_model, contrasts


def save_t_surfs(first_level_surf_model, contrasts, output_dir, prefix=""):
    # Contrast estimation and plotting
    print("\nContrast Estimation:")
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' %
              (index + 1, len(contrasts), contrast_id))
        t_map = first_level_surf_model.compute_contrast(contrast_val, output_type='stat')

        # Save the p map
        filename = op.join(output_dir, "{}{}.gii".format(prefix, contrast_id))
        t_map.to_filename(filename)
        print("\tT-map saved to : {}".format(filename))

        # Create snapshots of the contrasts
        # display = plotting.plot_stat_map(p_map, display_mode='z', threshold=0, title=contrast_id)


def glm(sub_dir, sub, run, contrasts_list_file=None, motion_file=None, lr_prefix=["lh_", "rh_"],
        tr=0.955, nbr_slice_by_tr=12):
    """Run surface GLM for hemispheres right and left with data projected on surface.

    :param sub_dir: Subject directory
    :param sub: Subject name
    :param run: Run name (or session name) Real name must start by prefixes 'lh_' or 'rh_' but
    the name given here doesn't contains this first character. For example, if there are files
    'lh_run_x.gii' and 'rh_run_x.gii, give only 'run_x'.
    :param contrasts_list_file: Filename of the text file that contains used contrasts
    :param motion_file: Path to motion regressors file.
    :param lr_prefix: prefix list to had to the run name to get .gii files names.
    :param tr: Repetition time
    :param nbr_slice_by_tr: Number of slice by TR
    :return: Save T-stat surface for left and right hemishperes.
    """
    slice_time_ref = tr / nbr_slice_by_tr
    paradigm_file = op.join(sub_dir, sub, "paradigms", run + '.csv')

    for prefix in lr_prefix:
        print("Process {} side".format(prefix))

        # Projected fMRI data
        fmri_img = op.join(sub_dir, sub, 'projection', '{}{}.gii'.format(prefix, run))
        output_dir = op.join(sub_dir, sub, 'output', run, 'glm_surf')

        first_level_surf_model, cntrst = run_glm(tr, slice_time_ref, paradigm_file,
                                                 fmri_img, motion_file)

        if contrasts_list_file is not None:
            # Read contrasts list defined in the text file
            contrasts_def = parse_contrasts_file(contrasts_file)
            cntrst = get_contrasts_list(contrasts_def, regressors_contrasts=cntrst)

        save_t_surfs(first_level_surf_model, cntrst, output_dir, prefix)


if __name__ == "__main__":
    if len(sys.argv) < 4 :
        print("Run GLM on surface data for the Localizer task.\n\n*** Args ***")
        print("  [1] Subject directory\n  [2] Subject's name\n  [3] Run name")
        print("  [4] Contrast definitio file\n  [5] (opt.) Motion file")
        exit(0)

    subject_dir = sys.argv[1]
    subject = sys.argv[2]
    run_name = sys.argv[3]
    contrasts_file = sys.argv[4]
    if len(sys.argv) > 5:
        motion_file = sys.argv[5]
    else:
        motion_file= None

    print("\n*** GLM on surface *********************************************")
    print("Subject dir: {}\nSubject: {}\nRun: {}".format(subject_dir, subject, run_name))
    print("Contrast file: {}\nMotion file: {}\n".format(contrasts_file, motion_file))

    glm(subject_dir, subject, run_name, contrasts_file, motion_file)
