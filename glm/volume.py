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
from nilearn import plotting

from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import make_design_matrix, plot_design_matrix


# **** Prepare data and analysis parameters ****
sub = "sub-01"

# Timing variables
tr = 0.955
slice_time_ref = tr/12

# Output directory. T-map of each contrast will be saved here.
output_dir = "/hpc/banco/bastien.c/data/fake_bids/sub-01/output/glm/"

# Read paradigm file (design)
paradigm_file = "/hpc/banco/InterTVA/bastien/sub-01/Sub-01_func01_SequenceLocalizer_Best_17_01_25_14_41_without_ISI.csv"
paradigm = pd.read_csv(paradigm_file, sep=',', index_col=None)

# Read motion regressor (.txt)
motion_file = "/hpc/banco/InterTVA/virginia/analyse_pilot/sub-01/func/session1/rp_sub-01_task-localizer-best_bold.txt"
motion_cols = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
motion = pd.read_csv(motion_file, sep="  ", index_col=None, names=motion_cols, engine='python')
nbr_vol = len(motion[motion.keys()[0]])
print("Number of scans in motion file: {}".format(nbr_vol))

# Read fMRI data
fmri_img = "/hpc/banco/InterTVA/virginia/analyse_pilot/sub-01/func/session1/" \
           "swusub-01_task-localizer-best_bold.nii"


# **** Build the design matrix ****
# Frame timing is determined with number of scans registered in the motion file
t_vect = np.arange(nbr_vol) * tr
# Create the design matrix without motion drift regressors
d_matrix = make_design_matrix(t_vect, paradigm=paradigm, drift_model='blank')
# Add motion regressors
for col in motion_cols:
    for i, t in enumerate(t_vect):
        motion[col][t] = motion[col][i]
    d_matrix[col] = motion[col]

# Plot the design matrix
plot_design_matrix(d_matrix)


# **** Perform first level analysis ****
# Setup and fit GLM
first_level_model = FirstLevelModel(tr, slice_time_ref, hrf_model='glover + derivative')
first_level_model = first_level_model.fit(fmri_img, design_matrices=d_matrix)


# **** Estimate contrasts ****
# Specify the contrasts
design_matrix = first_level_model.design_matrices_[0]
contrast_matrix = np.eye(design_matrix.shape[1])
contrasts = dict([(column, contrast_matrix[i])
                  for i, column in enumerate(design_matrix.columns)])

# Short list of more relevant contrasts
contrasts = {
    "voice vs non-voice": (contrasts["speech"] + contrasts["emotional"] + contrasts['non_speech']
                           - contrasts['artificial'] - contrasts['emotional'] - contrasts['environmental']),
    # "voice": (contrasts["speech"] + contrasts["emotional"] + contrasts['non_speech']),
    # "non-voice": (contrasts['artificial'] + contrasts['emotional'] + contrasts['environmental']),
    # "speech vs. non-speech": (contrasts['artificial'] - contrasts['emotional'] - contrasts['environmental'])
}

# Contrast estimation and plotting
print("\nContrast Estimation:")
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    p_map = first_level_model.compute_contrast(contrast_val, output_type='stat')

    # Save the p map
    p_map.to_filename(op.join(output_dir, contrast_id + ".nii"))
    print("\tT-map saved to : {}".format(op.join(output_dir, contrast_id + ".nii")))

    # Create snapshots of the contrasts
    display = plotting.plot_stat_map(p_map, display_mode='z', threshold=0, title=contrast_id)

plotting.show()
plt.show()