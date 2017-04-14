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
import argparse

from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import make_design_matrix, plot_design_matrix
import nibabel as nb

from design_optimisation.contrast_definition import get_contrasts_list, \
    parse_contrasts_file


def run_glm(tr, slice_time_ref, paradigm_file, fmri_img, outdir,
            motion_file=None):
    design_matrix_png_filename = op.join(outdir, "design_matrix.png")
    design_matrix_npy_filename = op.join(outdir, "design_matrix")

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

    # **** Perform first level glm ****
    # Setup and fit GLM
    print("Run GLM")
    first_lvl_model = FirstLevelModel(tr, slice_time_ref,
                                      hrf_model='glover + derivative',
                                      verbose=2)
    first_lvl_model = first_lvl_model.fit(fmri_img,
                                          design_matrices=d_matrix)

    # **** Estimate contrasts ****
    # Specify the contrasts
    design_matrix = first_lvl_model.design_matrices_[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i]) for i, column in
                     enumerate(design_matrix.columns)])

    return first_lvl_model, contrasts


def save_t_vols(first_lvl_model, contrasts, outdir, prefix=""):
    # Contrast estimation and plotting
    print("\nContrast Estimation:")
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' %
              (index + 1, len(contrasts), contrast_id))
        t_map = first_lvl_model.compute_contrast(contrast_val,
                                                 output_type='stat')

        # Save the p map
        filename = op.join(outdir, prefix + contrast_id + ".nii")
        t_map.to_filename(filename)
        print("\tT-map saved to : {}".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a GLM on Volume")
    parser.add_argument("-subdir", dest="subject_dir", type=str,
                        help="Path to subject directory")
    parser.add_argument("-sub", dest="subject", type=str, help="Subject name")
    parser.add_argument("-run", dest="run_name", type=str, help="Run name")
    parser.add_argument("-in", dest="fmri", type=str, help="fMRI images file")
    parser.add_argument("-con", dest="contrasts_file", type=str,
                        help="Contrast definition file")
    parser.add_argument("-motion", dest="motion_file", type=str, default=None,
                        help="Motion regressors file")
    parser.add_argument("-tr", dest="tr", type=float, help="Reptition time")
    parser.add_argument("-nslice", dest="nslices", type=int,
                        help="Number of slices by TR")
    args = parser.parse_args()

    # Print some infos
    print("\n*** GLM on surface *********************************************")
    print("Subject dir: {}".format(args.subject_dir))
    print("Subject: {}".format(args.subject))
    print("Run: {}".format(args.run_name))
    print("Contrast file: {}".format(args.contrasts_file))
    print("Motion file: {}".format(args.motion_file))
    print("TR: {}\tSlices by TR: {}\n".format(args.tr, args.nslices))

    # Define some usefull variable inferred from arguments
    slice_time_ref = args.tr / args.nslices
    paradigm_file = op.join(args.subject_dir, args.subject, "paradigms",
                            args.run_name + '.tsv')
    output_dir = op.join(args.subject_dir, args.subject, 'output',
                         args.run_name, 'glm_vol')

    # Run the GLM
    first_level_model, cntrst = run_glm(args.tr, slice_time_ref, paradigm_file,
                                        args.fmri, output_dir,
                                        motion_file=args.motion_file)

    # If a contrast definition file is specified, create the needed contrasts
    # Else keep one contrast by condition (or regressor)
    if args.contrasts_file is not None:
        # Read contrasts list defined in the text file
        contrasts_def = parse_contrasts_file(args.contrasts_file)
        cntrst = get_contrasts_list(contrasts_def, regressors_contrasts=cntrst)

    # Compute T-map and save them in the ouput directory
    save_t_vols(first_level_model, cntrst, output_dir)

    plt.show()
