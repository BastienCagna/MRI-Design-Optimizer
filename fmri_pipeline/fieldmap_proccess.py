

from os import system
import os.path as op
import argparse
import json as js
import warnings as wn
from common import log_msg


def create_acqparams_file(indir, outdir, sub, log=None):
    # Create acquisition parameter file ****************************************
    log_msg(log, "\nReading json file (from session 01, direction 01)")
    json_file = op.join(indir, "{}_acq-topup01_dir-01_epi.json".format(sub))
    infos = js.load(open(json_file))
    dir1 = infos['PhaseEncodingDirection']
    readoutTime = infos['TotalReadoutTime']

    # Check that all acquisition are similar
    tmp_file = op.join(indir, "{}_acq-topup02_dir-01_epi.json".format(sub))
    tmp_info = js.load(open(tmp_file))
    if tmp_info['TotalReadoutTime'] != readoutTime:
        log_msg("Total readout times are different.", warning=True)
    if tmp_info['PhaseEncodingDirection'] != dir1:
        log_msg("Phase encoding directions are different.", warning=True)

    log_msg(log, "Phase encoding direction: {}\nTotal readout time: {}".format(
        dir1, readoutTime))

    acqparam_file = op.join(outdir, "acqparamsAP_PA.txt")
    log_msg(log, "Acquisition params file: {}".format(acqparam_file))
    txtfile = open(acqparam_file, "w")
    if dir1 == "j-":
        # Dir 1
        txtfile.write("0 -1 0 {}\n".format(readoutTime))
        txtfile.write("0 -1 0 {}\n".format(readoutTime))
        txtfile.write("0 -1 0 {}\n".format(readoutTime))
        # Dir 2
        txtfile.write("0 1 0 {}\n".format(readoutTime))
        txtfile.write("0 1 0 {}\n".format(readoutTime))
        txtfile.write("0 1 0 {}\n".format(readoutTime))
    elif dir1 == "j":
        # Dir 1
        txtfile.write("0 1 0 {}\n".format(readoutTime))
        txtfile.write("0 1 0 {}\n".format(readoutTime))
        txtfile.write("0 1 0 {}\n".format(readoutTime))
        # Dir 2
        txtfile.write("0 -1 0 {}\n".format(readoutTime))
        txtfile.write("0 -1 0 {}\n".format(readoutTime))
        txtfile.write("0 -1 0 {}\n".format(readoutTime))
    else:
        wn.warn("Unrecognized phase encoding direction.")
        exit(-1)
    txtfile.close()
    return acqparam_file


def run_fsl(indir, outdir, sub, acqparam_file, fsl="fsl5.0", sessions=[1, 2],
            log=None):
    # Field maps ***************************************************************
    # Step 1: Merge 2 direction (AP-PA)
    # Setp 2: Apply top-up FSL
    log_msg(log, "\nField maps processing")

    # Session 1 & 2
    for s in sessions:
        log_msg(log, "\nSession {}".format(s))
        fieldmap_dir1 = op.join(indir, "{}_acq-topup{:02d}_dir-01_epi.nii.gz"
                                .format(sub, s))
        fieldmap_dir2 = op.join(indir, "{}_acq-topup{:02d}_dir-02_epi.nii.gz"
                                .format(sub, s))
        fieldmap = op.join(outdir, "fieldmap{:02d}".format(s))

        out = op.join(outdir, "topup_results{:02d}".format(s))
        fout = op.join(outdir, "field{:02d}".format(s))
        iout = op.join(outdir, "unwarped{:02d}".format(s))
        cmd = '{}-fslmerge -t {} {} {}'.format(fsl, fieldmap, fieldmap_dir1,
                                               fieldmap_dir2)
        log_msg(log, cmd)
        system(cmd)

        cmd = "{}-topup  --imain={} --datain={} --config=b02b0.cnf "\
              "--out={} --fout={} --iout={}".format(fsl, fieldmap,
                                                    acqparam_file, out, fout,
                                                    iout)
        log_msg(log, cmd)
        system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do fMRI data analyse")
    parser.add_argument("-indir", dest="indir", type=str,
                        help="Input subject directory")
    parser.add_argument("-outdir", dest="outdir", type=str,
                        help="Output subject directory")
    parser.add_argument("-sub", dest="sub", type=str, help="Subject name")
    args = parser.parse_args()

    # Path variables
    sub = args.sub
    indir = op.join(args.indir, sub, "fmap")
    outdir = op.join(args.outdir, sub, "fmap")

    # Create output dir
    print("\nCheck output directory")
    if not op.isdir(outdir):
        if not op.isdir(op.join(args.outdir, sub)):
            cmd = "mkdir {}".format(op.join(args.outdir, sub))
            print(cmd)
            system(cmd)
        cmd = "mkdir {}".format(outdir)
        print(cmd)
        system(cmd)

    # Create a log file
    log = open(op.join(outdir, 'log.txt'), 'w')

    acqp_file = create_acqparams_file(indir, outdir, sub, log)

    run_fsl(indir, outdir, sub, acqp_file)

    log.close()
