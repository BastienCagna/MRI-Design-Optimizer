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
# import seaborn as sns
import warnings
import numpy as np
import pandas as pd
import os.path as op
import sys

from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import make_design_matrix
import nibabel as nb

from design_optimisation.contrast_definition import get_contrasts_list, \
    parse_contrasts_file


def run_glm(tr, slice_time_ref, paradigm_file, fmri_img, output_dir,
            motion_file=None):
    design_matrix_png_filename = op.join(output_dir, "design_matrix.png")
    design_matrix_npy_filename = op.join(output_dir, "design_matrix")

    paradigm = pd.read_csv(paradigm_file, sep='\t', index_col=None)

    # Find the number of voulume in the .nii file
    nii_imgs = nb.load(fmri_img)
    nbr_vol = nii_imgs.get_data().shape[3]
    del nii_imgs
    print("Number of scans : {}".format(nbr_vol))

    # **** Build the design matrix ****
    if motion_file is not None:
        motion_cols = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
        motion = pd.read_csv(motion_file, sep="  ", index_col=None,
                             names=motion_cols, engine='python')

        if len(motion[motion.keys()[0]]) != nbr_vol:
            warnings.warn("Number of scan and row countof the motion file "
                          "dismatch.")
            exit(-1)

        # Frame timing is determined with number of scans registered in the
        # motion file
        t_vect = np.arange(nbr_vol) * tr
        # Create the design matrix without motion drift regressors
        d_matrix = make_design_matrix(t_vect, paradigm=paradigm,
                                      drift_model='blank')
        # Add motion regressors
        for col in motion_cols:
            for i, t in enumerate(t_vect):
                motion[col][t] = motion[col][i]
            d_matrix[col] = motion[col]
    else:
        t_vect = tr * np.arange(nbr_vol)
        d_matrix = make_design_matrix(t_vect, paradigm=paradigm,
                                      drift_model='blank')

    # Plot the design matrix
    x = d_matrix.as_matrix()
    np.save(design_matrix_npy_filename, x)
    print("Design matrix save as: {}".format(design_matrix_npy_filename))

    fig = plt.figure()
    plt.imshow(x, aspect="auto", interpolation="nearest")
    fig.savefig(design_matrix_png_filename)
    print("Design matrix save as: {}".format(design_matrix_png_filename))
    # plot_design_matrix(d_matrix)

    # **** Perform first level glm ****
    # Setup and fit GLM
    print("Run GLM")
    first_level_model = FirstLevelModel(tr, slice_time_ref,
                                        hrf_model='glover + derivative',
                                        verbose=2)
    first_level_model = first_level_model.fit(fmri_img, design_matrices=d_matrix)

    # **** Estimate contrasts ****
    # Specify the contrasts
    design_matrix = first_level_model.design_matrices_[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    cntrst = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns)])

    return first_level_model, cntrst


def save_t_vols(first_level_model, contrasts, output_dir):
    # Contrast estimation and plotting
    print("\nContrast Estimation:")
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' %
              (index + 1, len(contrasts), contrast_id))
        t_map = first_level_model.compute_contrast(contrast_val,
                                                   output_type='stat')

        # Save the p map
        t_map.to_filename(op.join(output_dir, "con_" + contrast_id + ".nii"))
        print("\tT-map saved to : {}".format(
            op.join(output_dir, "con_" + contrast_id + ".nii")))


def glm(sub_dir, sub, run, fmri_img, paradigm_file, contrasts_list_file=None,
        motion_file=None, tr=0.955, nbr_slice_by_tr=12):
    """

    :param sub_dir:
    :param sub:
    :param run:
    :param contrasts_list_file:
    :param fmri_img:
    :param paradigm_file:
    :param motion_file:
    :param tr:
    :param nbr_slice_by_tr:
    :return:
    """
    slice_time_ref = tr / nbr_slice_by_tr

    # Projected fMRI data
    output_dir = op.join(sub_dir, sub, 'output', 'glm_vol', run)

    first_level_model, cntrst = run_glm(tr, slice_time_ref, paradigm_file,
                                        fmri_img, output_dir,
                                        motion_file=motion_file)

    if contrasts_list_file is not None:
        # Read contrasts list defined in the text file
        contrasts_def = parse_contrasts_file(contrasts_list_file)
        cntrst = get_contrasts_list(contrasts_def, regressors_contrasts=cntrst)

    save_t_vols(first_level_model, cntrst, output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Run GLM on volumic data for the Localizer task.\n\n*** Args ***")
        print("  [1] Subject directory\n  [2] Subject's name\n  [3] Run name")
        print("  [4] Input nifti image\n  [5] Contrast definition file")
        print("  [6] (opt.) Motion file")
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

    plt.show()