import os.path as op
import numpy as np
import nibabel as nb
from nilearn.decoding import SearchLight
from sklearn.linear_model import LogisticRegression
import pandas as pd


subdir = "/hpc/banco/InterTVA/data"
sub = "sub-05"
task = "localizerbest"

data_pfx = 'u'
fmri_img = "{}/{}/func/{}{}_task-{}_bold_avg.nii".format(subdir, sub, data_pfx,
                                                         sub, task)
mask_file = "{}/{}/func/{}{}_task-{}_bold_mask.nii".format(
    subdir, sub, data_pfx, sub, task)

paradigm_file = "{}/{}/func/{}_task-{}_model-mvpa_events.tsv".format(
    subdir, sub, sub, task)


paradigm = pd.read_csv(paradigm_file, sep="\t")
y = paradigm['trial_type']

clf = SearchLight(nb.load(mask_file), verbose=50, radius=2,
                  estimator=LogisticRegression)
clf.fit(nb.load(fmri_img), y)
print(clf.scores_)
np.save("/hpc/banco/InterTVA/data/sub-05/output/sl_score_svc.npy", clf.scores_)
