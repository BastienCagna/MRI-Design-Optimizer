from glm.volume_00_no_rp import glm
import matplotlib.pyplot as plt
import numpy as np


def run_glm(sub_dir, sub_name, run_name, pfx_data, contrast_file, t_r):
    """Run nistats GLM on volumic data

    :param sub_dir: Subjects directory
    :param sub_name: Subject name
    :param run_name: Run name (or task)
    :param pfx_data: Prefix of the .nii file that specify the type of data used
    :param contrast_file: File that describe the contrast set
    :param t_r: Repetition time
    :return:
    """
    run_name = "{}{}_task-{}_nistats".format(pfx_data, sub_name, run_name)
    fmri_img = "{}/{}/func/{}{}_task-{}_bold.nii".format(
        sub_dir, sub_name, pfx_data, sub_name, run_name)
    paradigm_file = "{}/{}/func/{}_task-{}_model-glm_events.tsv".format(
        sub_dir, sub_name, sub_name, run_name)
    motion_file = "{}/{}/func/rp_{}_task-{}_bold.txt".format(
        sub_dir, sub_name, sub_name, run_name)
    print("Subject dir: {}\nSubject: {}".format(sub_dir, sub_name))

    glm(sub_dir, sub_name, run_name, fmri_img, paradigm_file,
        contrasts_list_file=contrast_file, tr=t_r, motion_file=motion_file)


if __name__ == "__main__":
    print("\n*** GLM on Volume ***********************************************")
    subject_dir = "/hpc/banco/InterTVA/data"
    prefix_dat = 'u'
    run = "localizerbest"
    task = "localizer"
    tr = 0.955
    con_file = "{}/sourcedata/paradigms/{}/contrasts_def_glm.txt".format(
        subject_dir, task)

    subjects_nbr = np.arange(0, 1)
    print("Subject numbers: {}".format(subjects_nbr))
    nbr_sub = len(subjects_nbr)

    print("TR: {}\nTask: {}\nContrast file: {}\n".format(tr, run, con_file))
    for i, n in enumerate(subjects_nbr):
        sub = "sub-{:02d}".format(n)
        print("\n** {}/{} {} ** ".format(i+1, nbr_sub, sub))
        run_glm(subject_dir, sub, run, prefix_dat, con_file, tr)



plt.show()
