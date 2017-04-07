from glm.volume_00_no_rp import glm
import matplotlib.pyplot as plt
import os.path as op

print("\n*** GLM on Volume ***************************************************")
output_dir_suffix = "_00_no_rp"
subject_dir = "/hpc/banco/bastien.c/data/inter_tva/"
con_file = op.join(subject_dir, "sourcedata/paradigms/localizer/"
                                "contrasts_localizer.txt")
prefix_dat = 'swu'

datasets = [
   {
      "subject_name": "pilote-01",
      "run_name": prefix_dat + "task-localizer_bold",
      "fmri_file": "/hpc/banco/InterTVA/virginia/analyse_pilot/S00/"
                   "Functional/Localizer/new/swutask-localizer_bold.nii",
      "paradigm_file": op.join(subject_dir, "pilote-01/paradigms/"
                                            "utask-localizer_bold.tsv"),
      "tr": 0.975
   },
   {
      "subject_name": "sub-01",
      "run_name": prefix_dat + "sub-01_task-localizer-best_bold",
      "fmri_file": "/hpc/banco/InterTVA/virginia/analyse_pilot/sub-01/"
                   "func/session1/" + prefix_dat +
                   "sub-01_task-localizer-best_bold.nii",
      "paradigm_file": op.join(subject_dir,
                               "sub-01/paradigms/"
                               "usub-01_task-localizer-best_bold.tsv"),
      "tr": 0.955
   },
   {
      "subject_name": "sub-02",
      "run_name": prefix_dat + "sub-02_task-localizer-best_bold",
      "fmri_file": "/hpc/banco/InterTVA/virginia/analyse_pilot/sub-02/"
                   "func/" + prefix_dat + "sub-02_task-localizer-best_bold.nii",
      "paradigm_file": op.join(subject_dir,
                               "sub-02/paradigms/"
                               "sub-02_task-localizer-best.tsv"),
      "tr": 0.955
   },
]

Nsets = len(datasets)
for i, dataset in enumerate(datasets):
    print("\n** {}/{} {} ** ".format(i+1, Nsets, dataset["subject_name"]))

    print("Subject dir: {}\nSubject: {}".
          format(subject_dir, dataset['subject_name']))
    print("Run: {}\n".format(dataset['run_name'] + output_dir_suffix))
    print("Contrast file: {}".format(con_file))

    glm(subject_dir, dataset['subject_name'], dataset['run_name'] +
        output_dir_suffix, dataset['fmri_file'], dataset['paradigm_file'],
        contrasts_list_file=con_file, tr=dataset['tr'])


plt.show()
