import nilearn as ni
import nibabel as nb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


subdir = "/hpc/banco/InterTVA/data/"
sub = "sub-05"
data_prf = 'u'
task = 'localizerbest'
tr = 0.955


# First frame taken in account after an onset (0 is the first)
frame_offset = 2
# Last frame taken in account after an onset (the offset in no taken in account)
frame_end = 8


nii_file = "{}/{}/func/{}{}_task-{}_bold.nii".format(subdir, sub, data_prf,
                                                     sub, task)
paradigm_file = "{}/{}/func/{}_task-{}_model-mvpa_events.tsv".format(subdir,
                                                                    sub, sub,
                                                              task)
output_nii = "{}/{}/func/{}{}_task-{}_bold_avg.nii".format(subdir, sub,
                                                           data_prf, sub, task)


# Load the Nifti file and convert it to a 2D matrix that contains one image
# by row
nii_img = nb.load(nii_file)
u = nii_img.shape[0]
v = nii_img.shape[1]
ncoupes = nii_img.shape[2]
nvol = nii_img.shape[3]

n_features = u * v * ncoupes
imgs = np.array(nii_img.dataobj)

# Load the paradigm file to get onsets and corresponding y value (trial_type)
paradigm = pd.read_csv(paradigm_file, sep="\t")

onsets = paradigm['onset']
y = paradigm['trial_type']
nbr_events = len(y)

# Construct the x matrix by averaging images that follow an onsets
x = np.zeros((u, v, ncoupes, nbr_events))
for i, onset in enumerate(onsets):
    # Use ceil() to be sure to get images acquired after the onset
    k = int(np.ceil(onset / tr))
    x[:, :, :, i] = np.mean(imgs[:, :, :, k + frame_offset:k + frame_end],
                           axis=3)

# outdata = np.reshape(x, (u, v, ncoupes, nbr_events))
imgs = nb.Nifti1Image(x, nii_img.affine)
nb.save(imgs, output_nii)
