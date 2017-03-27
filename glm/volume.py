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

from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import make_design_matrix, plot_design_matrix
import nibabel as nb

from design_optimisation.contrast_definition import get_contrasts_list, parse_contrasts_file


def run_glm(tr, slice_time_ref, paradigm_file, fmri_img, motion_file=None):
    paradigm = pd.read_csv(paradigm_file, sep=',', index_col=None)

    # **** Build the design matrix ****
    if motion_file is not None:
        motion_cols = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
        motion = pd.read_csv(motion_file, sep="  ", index_col=None, names=motion_cols, engine='python')
        nbr_vol = len(motion[motion.keys()[0]])
        print("Number of scans in motion file: {}".format(nbr_vol))

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
        nii_imgs = nb.read(fmri_img)
        nbr_vol = len(nii_imgs.darrays)
        del nii_imgs
        print("Number of scans : {}".format(nbr_vol))
        t_vect = tr * np.arange(nbr_vol)
        d_matrix = make_design_matrix(t_vect, paradigm=paradigm, drift_model='blank')


    # Plot the design matrix
    plot_design_matrix(d_matrix)

    # **** Perform first level analysis ****
    # Setup and fit GLM
    first_level_model = FirstLevelModel(tr, slice_time_ref, hrf_model='glover + derivative',
                                        verbose=2)
    first_level_model = first_level_model.fit(fmri_img, design_matrices=d_matrix)

    # **** Estimate contrasts ****
    # Specify the contrasts
    design_matrix = first_level_model.design_matrices_[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    cntrst = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns)])

    return first_level_model, cntrst


def save_t_vols(first_level_model, contrasts, output_dir, prefix=""):
    # Contrast estimation and plotting
    print("\nContrast Estimation:")
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' %
              (index + 1, len(contrasts), contrast_id))
        p_map = first_level_model.compute_contrast(contrast_val, output_type='stat')

        # Save the p map
        p_map.to_filename(op.join(output_dir, prefix + contrast_id + ".nii"))
        print("\tT-map saved to : {}".format(op.join(output_dir, prefix + contrast_id + ".nii")))

        # Create snapshots of the contrasts
        # display = plotting.plot_stat_map(p_map, display_mode='z', threshold=0, title=contrast_id)

#
# def localizer_best_glm():
#     # **** Prepare data and analysis parameters ****
#     sub = "sub-01"
#
#     # Timing variables
#     tr = 0.955
#     slice_time_ref = tr/12
#
#
#     # Output directory. T-map of each contrast will be saved here.
#     output_dir = "/hpc/banco/bastien.c/data/fake_bids/sub-01/output/localizer-best/glm_vol"
#
#     # Read paradigm file (design)
#     paradigm_file = "/hpc/banco/InterTVA/bastien/sub-01/" \
#                     "Sub-01_func01_SequenceLocalizer_Best_17_01_25_14_41_glm.csv"
#
#     # Read motion regressor (.txt)
#     motion_file = "/hpc/banco/InterTVA/virginia/analyse_pilot/sub-01/func/session1/" \
#                   "rp_sub-01_task-localizer-best_bold.txt"
#
#
#     # Read fMRI data
#     fmri_img = "/hpc/banco/InterTVA/virginia/analyse_pilot/sub-01/func/session1/" \
#                "swusub-01_task-localizer-best_bold.nii"
#
#     first_level_model, cntrst = run_glm(tr, slice_time_ref, paradigm_file, motion_file, fmri_img)
#
#     # Short list of more relevant contrasts
#     cntrst = {
#         "voice_minus_non-voice": (cntrst["speech"] + cntrst["emotional"] + cntrst['non_speech']
#                                - cntrst['artificial'] - cntrst['emotional'] - cntrst[
#                                    'environmental']),
#         "voice": (cntrst["speech"] + cntrst["emotional"] + cntrst['non_speech']),
#         "non-voice": (cntrst['artificial'] + cntrst['emotional'] + cntrst['environmental']),
#         "speech_minus_non-speech": (cntrst["speech"] - cntrst['non_speech']),
#         "all_minus_silence": (cntrst["speech"] + cntrst["emotional"] + cntrst['non_speech']
#                                + cntrst['artificial'] + cntrst['emotional'] + cntrst['environmental'])
#     }
#
#     save_t_vols(first_level_model, cntrst, output_dir)
#
#
# def localizer_moyen_glm():
#     # **** Prepare data and analysis parameters ****
#     sub = "sub-01"
#
#     # Timing variables
#     tr = 0.955
#     slice_time_ref = tr/12
#
#     # Output directory. T-map of each contrast will be saved here.
#     output_dir = "/hpc/banco/bastien.c/data/fake_bids/sub-01/output/localizer-moyen/glm_vol"
#
#     # Read paradigm file (design)
#     paradigm_file = "/hpc/banco/InterTVA/bastien/sub-01/" \
#                     "Sub-01_func02_SequenceLocalizer_Moy_17_01_25_14_56_glm.csv"
#
#     # Read motion regressor (.txt)
#     motion_file = "/hpc/banco/InterTVA/virginia/analyse_pilot/sub-01/func/session1/" \
#                   "rp_sub-01_task-localizer-moyen_bold.txt"
#
#     # Read fMRI data
#     fmri_img = "/hpc/banco/InterTVA/virginia/analyse_pilot/sub-01/func/session1/" \
#                "swusub-01_task-localizer-moyen_bold.nii"
#
#     first_level_model, cntrst = run_glm(tr, slice_time_ref, paradigm_file, motion_file, fmri_img)
#
#     # Short list of more relevant contrasts
#     cntrst = {
#         "voice_minus_non-voice": (cntrst["speech"] + cntrst["emotional"] + cntrst['non_speech']
#                                - cntrst['artificial'] - cntrst['emotional'] - cntrst[
#                                    'environmental']),
#         "voice": (cntrst["speech"] + cntrst["emotional"] + cntrst['non_speech']),
#         "non-voice": (cntrst['artificial'] + cntrst['emotional'] + cntrst['environmental']),
#         "speech_minus_non-speech": (cntrst["speech"] - cntrst['non_speech']),
#         "all_minus_silence": (cntrst["speech"] + cntrst["emotional"] + cntrst['non_speech']
#                                + cntrst['artificial'] + cntrst['emotional'] + cntrst['environmental'])
#     }
#
#     save_t_vols(first_level_model, cntrst, output_dir)


def glm(sub_dir, sub, run, fmri_img, contrasts_list_file=None, motion_file=None,
        tr=0.955, nbr_slice_by_tr=12):
    """

    :param sub_dir:
    :param sub:
    :param run:
    :param contrasts_list_file:
    :param fmri_img:
    :param motion_file:
    :param tr:
    :param nbr_slice_by_tr:
    :return:
    """
    slice_time_ref = tr / nbr_slice_by_tr
    paradigm_file = op.join(sub_dir, sub, "paradigms", run + '.csv')

    # Projected fMRI data
    output_dir = op.join(sub_dir, sub, 'output', run, 'glm_vol')

    first_level_model, cntrst = run_glm(tr, slice_time_ref, paradigm_file, fmri_img,
                                        motion_file)

    if contrasts_list_file is not None:
        # Read contrasts list defined in the text file
        contrasts_def = parse_contrasts_file(contrasts_file)
        cntrst = get_contrasts_list(contrasts_def, regressors_contrasts=cntrst)

    save_t_vols(first_level_model, cntrst, output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Run GLM on volumic data for the Localizer task.\n\n*** Args ***")
        print("  [1] Subject directory\n  [2] Subject's name\n  [3] Run name")
        print("  [4] Input nifti image\n  [5] Contrast definition file\n  [6] (opt.) Motion file")
        exit(0)

    subject_dir = sys.argv[1]
    subject = sys.argv[2]
    run_name = sys.argv[3]
    fmri = sys.argv[4]
    contrasts_file = sys.argv[5]
    if len(sys.argv) > 6:
        motion = sys.argv[6]
    else:
        motion = None

    print("\n*** GLM on surface *********************************************")
    print("Subject dir: {}\nSubject: {}\nRun: {}".format(subject_dir, subject, run_name))
    print("Contrast file: {}\nMotion file: {}\n".format(contrasts_file, motion))

    glm(subject_dir, subject, run_name, fmri, contrasts_file, motion, tr=0.975)

    # localizer_moyen_glm()
    # plotting.show()
    plt.show()