import argparse
from common import log_msg
from fmri_pipeline.fieldmap_proccess import create_acqparams_file, run_fsl
import os.path as op
from os import system


def run_cmd(command, log_f=None):
    if log_f is not None:
        log_msg(log_f, command)
    else:
        print(command)
    system(command)


def check_directories(directory, subject):
    """Create directory if subjects and/or subject directories doesn't exist.

    :param directory: Subjects directory
    :param subject: Subject name
    """
    print("\nCheck output directory")
    if not op.isdir(op.join(directory, subject)):
        if not op.isdir(directory):
            run_cmd("mkdir {}".format(directory))
        run_cmd("mkdir {}".format(op.join(directory, subject)))


def print_step_action(log, arg, msg):
    if arg is True:
        txt = '\t\t[YES] '
    else:
        txt = '\t\t[NO]  '

    log_msg(log, txt+msg)


def print_steps(log_f, arg_list):
    log_msg(log_f, "\nPipeline steps:")
    log_msg(log_f, "\tPreprocessing:")
    print_step_action(log_f, arg_list.doCopy, "Copy original files")
    print_step_action(log_f, arg_list.doHcp, "Run HCP preprocessing pipeline")
    # print_step_action(log_f, arg_list.doFmap, "Create fieldmap")
    print_step_action(log_f, arg_list.doSpm, "Run SPM fMRI preprocessing")
    log_msg(log_f, "\tGLM:")
    print_step_action(log_f, arg_list.doConvEvent, "Convert labview output "
                                                   "files to paradigm files")
    print_step_action(log_f, arg_list.doVolGLM, "Run GLM on volume using SPM")
    print_step_action(log_f, arg_list.doSurfGLM, "Run GLM on surface using "
                                                 "Nistats")


def create_fieldmap(indir, outdir, sub_name, log_f):
    """Create fieldmap using FSL

    :param indir: Original fmap directory
    :param outdir: Target fmap directory
    :param sub_name: Subject name
    :param log_f: Log file
    :return:
    """
    acqp_file = create_acqparams_file(indir, outdir, sub_name, log_f)
    run_fsl(indir, outdir, sub_name, acqp_file)


def copy_files(log_f, source_dir, target_dir, subj):
    source_dir = op.join(source_dir, subj)
    target_dir = op.join(target_dir, subj)

    log_msg(log_f, "Create subject directories", print_time=True,
            blank_line=True)
    run_cmd("mkdir {}/anat/".format(target_dir), log_f)
    run_cmd("mkdir {}/fmap/".format(target_dir), log_f)
    run_cmd("mkdir {}/func/".format(target_dir), log_f)
    run_cmd("mkdir {}/output/".format(target_dir), log_f)
    run_cmd("mkdir {}/output/glm_vol/".format(target_dir), log_f)

    log_msg(log_f, "Copy functional files", print_time=True, blank_line=True)
    run_cmd("cp {}/func/* {}/func/".format(source_dir, target_dir), log_f)

    log_msg(log_f, 'Copy others files', print_time=True, blank_line=True)
    run_cmd("cp {}/anat/* {}/anat/".format(source_dir, target_dir), log_f)
    run_cmd("cp {}/fmap/{}_acq-topup1_fieldmap.nii.gz "
            "{}/fmap/{}_acq-topup1_fieldmap.nii.gz".format(source_dir, subj,
                                                           target_dir, subj),
            log_f)
    run_cmd("cp {}/fmap/{}_acq-topup1_magnitude.nii.gz "
            "{}/fmap/{}_acq-topup1_magnitude.nii.gz".format(
                    source_dir, subj, target_dir, subj), log_f)

    log_msg(log_f, 'Unzip .nii.gz files', print_time=True, blank_line=True)
    run_cmd("gunzip {}/fmap/*.nii.gz".format(target_dir), log_f)
    run_cmd("gunzip {}/anat/*.nii.gz".format(target_dir), log_f)
    run_cmd("gunzip {}/func/*.nii.gz".format(target_dir), log_f)


def hcp_minimal_preprocessing(log_f, sub_dir, subj, t1_file, t2_file):
    """Create surface using T1 and T2 images (HCP Pipeline)

    :param subj: Subject name
    :param hcp_outdir: Subject HCP output directory
    :param t1_file: Path to T1 image
    :param t2_file:  Path to T2 image
    :return: Results are stored in hcp_outdir/sub/ by the HCP pipeline.
    """
    scripts_dir = "{}/sourcedata/scripts/batch".format(sub_dir)
    hcp_stp = op.join(scripts_dir, "hcp_setup.sh")

    # Config
    log_msg(log, "** HCP setup **", print_time=True, blank_line=True)
    run_cmd("mkdir {}/hcp".format(op.join(sub_dir, sub)))

    # Pre-freesurfer
    log_msg(log, "** Pre-FreeSurfer **", print_time=True, blank_line=True)
    run_cmd("{}/hcp_pre-fs.sh {} {} {} {}".format(scripts_dir,
            sub_dir, subj, t1_file, t2_file), log_f)

    # Freesurfer
    log_msg(log, "** FreeSurfer **", print_time=True, blank_line=True)
    run_cmd("{} && {}/hcp_fs.sh {} {}".format(hcp_stp, scripts_dir, sub_dir,
                                              subj), log_f)

    # Post-freesurfer
    log_msg(log, "** Post-FreeSurfer **", print_time=True, blank_line=True)
    run_cmd("{} && {}/hcp_post-fs.sh {} {}".format(hcp_stp, scripts_dir,
                                                   sub_dir, subj), log_f)


def spm_fmri_preprocessing(sub_dir, subj, t1_file):
    script_dir = op.join(sub_dir, "sourcedata", "scripts")

    m_script = "sub='{}'; bids_dir='{}'; T1w_file='{}';spm fmri;".format(
        subj, sub_dir, t1_file)

    # T1 coreg + VDM computation
    m_script += "run('{}/matlab/spm_fmri_T1coreg_and_vdm.m'); " \
                "spm_jobman('run',matlabbatch); clear matlabbatch;".format(
                 script_dir)

    # Unwarping
    m_script += "run('{}/matlab/spm_fmri_preprocessing.m'); " \
                "spm_jobman('run',matlabbatch); clear matlabbatch;".format(
                 script_dir)

    # Coreg + norm + seg
    m_script += "run('{}/matlab/coreg_norm_seg.m'); " \
                "spm_jobman('run',matlabbatch); clear matlabbatch;".format(
                 script_dir)

    # Smoothing
    #m_script +=

    m_script += "exit;"
    run_cmd('matlab -r "{}"'.format(m_script))


def conv_event_file(log_f, src_dir, tgt_dir, subj):
    # TODO: create relative path or something not fixed
    script_path = "/hpc/banco/bastien.c/python/design_optimizer/tools" \
                  "/labview_output_to_paradigm_file.py"

    src_dir = op.join(src_dir, "sourcedata", subj)
    db_f_loc = op.join(tgt_dir, "sourcedata/paradigms/localizer/stim_db.tsv")
    db_f_id = op.join(tgt_dir, "sourcedata/paradigms/identification/"
                               "stim_db.tsv")
    db_f_cal = op.join(tgt_dir, "sourcedata/paradigms/voicecalibrator/"
                                "stim_db.tsv")
    tgt_dir = op.join(tgt_dir, subj)

    run_cmd("python {} {}/{}_func01 -db {} -outTemplate {}/func/{}_task-{} "
            "-ftype labview -quest".format(script_path, src_dir, sub, db_f_id,
                                           tgt_dir, sub,
                                           "identification_run-01"), log_f)
    run_cmd("python {} {}/{}_func02 -db {} -outTemplate {}/func/{}_task-{} "
            "-ftype labview -quest".format(script_path, src_dir, sub, db_f_id,
                                           tgt_dir, sub,
                                           "identification_run-01"), log_f)
    run_cmd("python {} {}/{}_func03 -db {} -outTemplate {}/func/{}_task-{} "
            "-ftype labview -quest".format(script_path, src_dir, sub, db_f_id,
                                           tgt_dir, sub,
                                           "identification_run-01"), log_f)
    run_cmd("python {} {}/{}_func04 -db {} -outTemplate {}/func/{}_task-{} "
            "-ftype labview -quest".format(script_path, src_dir, sub, db_f_id,
                                           tgt_dir, sub,
                                           "identification_run-01"), log_f)
    run_cmd("python {} {}/{}_func05 -db {} -outTemplate {}/func/{}_task-{} "
            "-ftype labview".format(script_path, src_dir, sub, db_f_loc,
                                    tgt_dir, sub, "localizerbest"), log_f)
    run_cmd("python {} {}/{}_func06 -db {} -outTemplate {}/func/{}_task-{} "
            "-ftype labview".format(script_path, src_dir, sub, db_f_cal,
                                    tgt_dir, sub, "voicecalibrator"), log_f)


def spm_glm_localizer(sub_dir, subj, data_pfx):
    script_dir = op.join(sub_dir, "sourcedata/scripts")
    m_script = "sub='{}'; subdir='{}'; data_prf='{}';".format(subj, sub_dir,
                                                              data_pfx)
    m_script += "spm fmri; run('{}/matlab/glm_localizer.m'); " \
                "spm_jobman('run',matlabbatch); exit;".format(script_dir)
    run_cmd('matlab -r "{}"'.format(m_script))


def delete_original_func_files(log_f, sdir, s):
    run_cmd("rm {}/func/{}*_bold.nii".format(op.join(sdir, s), s), log_f)
    run_cmd("rm {}/func/{}*_sbref.nii".format(op.join(sdir, s), s), log_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do fMRI data preprocessing "
                                                 "and GLM for first level "
                                                 "analysis.")
    parser.add_argument("-subdir", dest="subdir", type=str,
                        help="Subject directory")
    parser.add_argument("-indir", dest="indir", type=str, default=None,
                        help="Subject directory containing original data")
    parser.add_argument("-sub", dest="sub", type=str, help="Subject name")

    parser.add_argument("-t1name", dest="t1name", default="_T1w.nii",
                        help="T1 name will be: [sub][t1name]")

    # Process activation
    parser.add_argument("-all", dest="doAll", action='store_const',
                        const=True, default=False, help="Activate all steps.")
    parser.add_argument("-preproc", dest="doPreProc", action='store_const',
                        const=True, default=False, help="Activate all "
                                                        "preprocessing steps.")
    parser.add_argument("-glm", dest="doGlm", action='store_const', const=True,
                        default=False, help="Activate all GLM steps.")

    # Preprocessing
    parser.add_argument("-copy", dest="doCopy", action='store_const',
                        const=True, default=False,
                        help="Activate original data copying in subdir.")
    parser.add_argument("-hcp", dest="doHcp", action='store_const',
                        const=True, default=False,
                        help="Activate HCP Pipeline.")
    parser.add_argument("-fmap", dest="doFmap", action='store_const',
                        const=True, default=False,
                        help="Activate fieldmap creation")
    parser.add_argument("-spm", dest="doSpm", action='store_const',
                        const=True, default=False,
                        help="Activate full SPM fMRI preprocessing pipeline")
    # GLM
    parser.add_argument("-convEvent", dest="doConvEvent", action='store_const',
                        const=True, default=False, help="Activate paradigm "
                        "file creation from labview file")
    parser.add_argument("-volGLM", dest="doVolGLM", action='store_const',
                        const=True, default=False,
                        help="Activate GLM on volume using SPM")
    parser.add_argument("-surfGLM", dest="doSurfGLM", action='store_const',
                        const=True, default=False,
                        help="Activate GLM on surface using Nistats")
    parser.add_argument("-delOrigFunc", dest="doDelOrigFunc",
                        action='store_const', const=True, default=False,
                        help="Remove original functional files")
    args = parser.parse_args()

    # --- INIT -----------------------------------------------------------------
    # Path variables
    sub = args.sub
    subdir = args.subdir

    # Create subject dir
    check_directories(subdir, sub)

    # Create a log file
    log = open(op.join(subdir, sub, 'log.txt'), 'a')
    log_msg(log, "************************************************************"
                 "********************", blank_line=True)
    log_msg(log, "New call to the python pipeline.\n", print_time=True)

    # --- CONFIG ---------------------------------------------------------------
    # Check what must be done
    if args.doAll:
        args.doPreProc = True
        args.doGlm = True
    if args.doPreProc:
        args.doCopy = True
        args.doHcp = True
        args.doSpm = True
    if args.doGlm:
        args.doConvEvent = True
        args.doVolGLM = True
        args.doSurfGLM = True

    print_steps(log, args)

    # Usefull path
    log_msg(log, "\nFixed files and directories:")
    t1_file = "{}/{}/anat/{}{}".format(subdir, sub, sub, args.t1name)
    t2_file = "{}/{}/anat/{}_T2w.nii".format(subdir, sub, sub)
    log_msg(log, "\tSubjects directory; {}".format(subdir))
    log_msg(log, "\tT1 file: {}\n\tT2 file: {}".format(t1_file, t2_file))

    # --- WORK -----------------------------------------------------------------
    # Copy original data
    if args.doCopy is True:
        log_msg(log, "*** Files copy and sub directory creation",
                blank_line=True, print_time=True)
        copy_files(log, args.indir, subdir, sub)

    # Go to the output directory to be sur that new files create in the
    # current directory will be in that directory
    run_cmd("cd {}/output".format(op.join(subdir, sub)), log)

    # HCP
    if args.doHcp is True:
        log_msg(log, "*** HCP Pipeline", blank_line=True, print_time=True)
        hcp_outdir = op.join(subdir, sub, "hcp")
        hcp_minimal_preprocessing(log, subdir, sub, t1_file, t2_file)

    # Fieldmap
    # if args.doFmap:
    #     log_msg(log, "*** Fieldmap (FSL)", blank_line=True, print_time=True)
    #     create_fieldmap()

    # SPM
    if args.doSpm:
        log_msg(log, "*** SPM Preprocessing", blank_line=True, print_time=True)
        spm_fmri_preprocessing(subdir, sub, t1_file)

    # Paradigm files
    if args.doConvEvent:
        log_msg(log, "*** Paradigm file creation", blank_line=True,
                print_time=True)
        conv_event_file(log, args.indir, subdir, sub)

    # GLM on volume with SPM
    if args.doVolGLM:
        data_pfx_glm = 'wu'
        log_msg(log, "*** GLM on volume (SPM)", blank_line=True,
                print_time=True)
        log_msg(log, "Data type:{}".format(data_pfx_glm))
        spm_glm_localizer(subdir, sub, data_pfx_glm)

    # GLM on surface with nistats
    if args.doSurfGLM:
        log_msg(log, "*** GLM on surface (nistats)", blank_line=True,
                print_time=True)
        log_msg(log, "Nothing to do", warning=True)

    # --- CLOSE ----------------------------------------------------------------
    log_msg(log, "")
    log_msg(log, "All is done. Bye.", print_time=True, blank_line=True)
    log_msg(log, "************************************************************"
                 "********************")
    log.close()
