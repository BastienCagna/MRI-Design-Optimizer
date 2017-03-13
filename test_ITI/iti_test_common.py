import numpy as np


def set_fixed_iti(design, const_iti=0):
    new_design = dict(design)
    for i in range(len(design['onset'])-1):
        new_design['onset'][i+1] = new_design['onset'][i] + new_design['duration'][i] + const_iti
        new_design['ITI'][i] = const_iti
    return new_design


def set_random_iti(design, iti_avg, iti_std):
    new_design = dict(design)
    nbr_event = len(new_design['onset'])
    iti_v = np.random.normal(iti_avg, iti_std, nbr_event)
    for i in range(nbr_event-1):
        new_design['onset'][i+1] = new_design['onset'][i] + new_design['duration'][i] + iti_v[i]
        new_design['ITI'][i] = iti_v[i]
    return new_design


def limited_duration(design, duration):
    i_max = np.where(design['onset']<=duration)[-1][-1]
    new_design = {}
    for k in design.keys():
        new_design[k] = design[k][:i_max]
    return new_design

