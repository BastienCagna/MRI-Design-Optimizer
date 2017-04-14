import numpy as np
import nibabel as nb
from nilearn.plotting import plot_roi
import matplotlib.pyplot as plt


subdir = "/hpc/banco/InterTVA/data"
sub = "sub-05"
task = "localizerbest"

data_pfx = 'u'
fmri_img = "{}/{}/func/{}{}_task-{}_bold.nii".format(subdir, sub, data_pfx,
                                                     sub, task)
mask_file = "{}/{}/func/{}{}_task-{}_bold_mask.nii".format(
    subdir, sub, data_pfx, sub, task)

imgs = nb.load(fmri_img)
affine = imgs.affine
imgs = np.array(imgs.dataobj)

var = np.var(imgs, axis=3)
avg = np.mean(imgs, axis=3)

mask = var > 10000
mask = np.array(mask, dtype=int)

print("{} voxel(s) are unvariant.".format(np.sum(mask)))

avg_nii = nb.Nifti2Image(avg, affine)
mask_nii = nb.Nifti2Image(mask, affine)
nb.save(mask_nii, mask_file)

plot_roi(mask_nii, avg_nii)

plt.show()