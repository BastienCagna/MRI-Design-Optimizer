"""
Searchlight analysis of voice vs nonvoice recognition
==================================================

Searchlight analysis requires fitting a classifier a large amount of
times. As a result, it is an intrinsically slow method. In order to speed
up computing, in this example, Searchlight is run only on one slice on
the fMRI (see the generated figures).

"""
import numpy as np
import nibabel
import nilearn.decoding

from nilearn.image import new_img_like, concat_imgs, index_img

import os.path as op
import glob

from sklearn.cross_validation import StratifiedKFold


def vl_searchlight(sub_dir, sub):
    # Results folder
    output_dir = op.join(sub_dir, sub, "output")

    glm_dir = "{}/wu{}_task*localizerbest_spm".format(output_dir, sub)

    # Create dataset for one subject with voice localizer
    vl_dataset = dict()
    vl_dataset['anat'] = glob.glob(op.join(sub_dir, sub,
                                           'anat', '{}_*_T1w.nii'.format(sub)))

    # take all the functional files thanks to glob
    vl_dataset['func'] = glob.glob(op.join(glm_dir, 'beta_*.nii'))

    # remove motion regressors and constant (the last 7 betas)
    vl_dataset['func'] = vl_dataset['func'][:-7]

    # Take the original mask (whole brain)
    vl_dataset['mask'] = op.join(glm_dir, 'mask.nii')

    # Take the mask of the temporal lobes (the same for all subjects)
    # vl_dataset['ROI'] = op.join(root_vl_datadir,'Bilateral_TVA.nii')

    # Take text files with labels
    vl_dataset['session_target'] = [op.join(output_dir, 'vl_labels.txt')]

    # print basic information on the dataset
    print('First subject anatomical nifti image (3D) is at: {}'.format(
          vl_dataset['anat'][0]))
    print('First subject functional nifti images (3D) are at: {}'.format(
          vl_dataset['func'])) # 4D data

    labels = np.recfromcsv(vl_dataset['session_target'][0], delimiter=" ")
    target = labels['labels']

    # Keep only data corresponding to voices or nonvoices
    condition_mask = np.logical_or(labels['labels'] == b'voice',
                                   labels['labels'] == b'nonvoice')
    target = target[condition_mask]
    y = target

    # Prepare masks
    # - mask_img is the original mask ( = full brain); in our case = mask_whole
    # - process_mask_img is a subset of mask_img, it contains the voxels that
    #   should be processed; in our case = mask_roi

    # For decoding, standardizing is often very important
    # 1-take the whole brain mask
    mask_img = nibabel.load(vl_dataset['mask'])

    # 2-take the ROI mask of left and right STG
    process_mask_img = nibabel.load(vl_dataset['mask'])

    fmri_img_list = []
    for fmri_filename in vl_dataset['func']:
        fmri_img_list.append(nibabel.load(fmri_filename))

    fmri_img = concat_imgs(fmri_img_list)

    # Restrict to only images for our decoding task
    fmri_img = index_img(fmri_img, condition_mask)

    # Searchlight computation

    # Make processing parallel
    # /!\ As each thread will print its progress, n_jobs > 1 could mess up the
    #     information output.
    n_jobs = 12

    # Define the cross-validation scheme used for validation.
    # Here we use a KFold cross-validation on the session, which corresponds to
    # splitting the samples in 4 folds and make 4 runs using each fold as a test
    # set once and the others as learning sets

    # from sklearn.cross_validation import StratifiedKFold
    # cv = StratifiedKFold(y.size, n_folds=10)

    cv = StratifiedKFold(target, n_folds=10)

    # The radius is the one of the Searchlight sphere that will scan the volume
    searchlight = nilearn.decoding.SearchLight(
        mask_img, process_mask_img=process_mask_img, radius=4, n_jobs=n_jobs,
        verbose=4, cv=cv)
    searchlight.fit(fmri_img, y)

    scores_img = new_img_like(mask_img, searchlight.scores_)
    cv_scores = np.mean(searchlight.scores_)
    print("Searchlight scores:\n{}".format(cv_scores))

    output_path = op.join(output_dir, 'slscores_vonv_tva_{}.nii.gz'.format(sub))
    scores_img.to_filename(output_path)

if __name__ == "__main__":
    subjects_dir = "/hpc/banco/InterTVA/data/"
    sub_nums = [6]
    for sub_num in sub_nums:
        vl_searchlight(subjects_dir, "sub-{:02d}".format(sub_num))
