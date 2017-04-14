import warnings as wn
import datetime


def log_msg(log_file, msg, warning=False, print_time=False, blank_line=False):
    if blank_line is True:
        txt = "\n"
    else:
        txt = ""

    if print_time is True:
        txt += datetime.datetime.now().strftime("%B %d %Y %H:%M:%S : ")
    if warning is True:
        txt += "/!\ Warning: " + msg
        wn.warn(msg)
    else:
        txt += msg

    if log_file is not None:
        log_file.write(txt + "\n")

    if warning is False:
        print(txt)
