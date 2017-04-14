from os import system
import numpy as np

subject_nums = np.arange(3, 8)

t1_names = [
    '_T1w.nii',
    '_T1w.nii',
    '_acq-01_T1w.nii',
    '_acq-01_T1w.nii',
    '_acq-01_T1w.nii',
]

for i, sub_num in enumerate(subject_nums):
    sub = "sub-{:02d}".format(sub_num)

    print(sub_num)

    system("/hpc/soft/anaconda3/bin/python "
           "/hpc/banco/bastien.c/python/design_optimizer/fmri_pipeline"
           "/first_level_pipeline.py -subdir /hpc/banco/InterTVA/data "
           "-indir /hpc/banco/InterTVA/bids_data -sub {} -all "
           "-t1name {}".format(sub, t1_names[i]))
