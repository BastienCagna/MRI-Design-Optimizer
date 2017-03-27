import numpy as np
import pandas as pd
import pickle
import time
from design_optimisation.random_design_creator import  add_answer_event


np_designs = np.load("/hpc/banco/bastien.c/data/optim/identification/final/designs.npy")
stim_db = pd.read_csv("/hpc/banco/bastien.c/data/optim/identification/stim_db.csv", sep='\t')
#i_v = [32562, 45929, 56492, 86180]

spk_names = {'182': 'Anne', '214': 'Betty', '233': 'Chloe'}

t = time.time()
designs = []
isi_maxs = []
for i in range(100000):
    if np.mod(i, 100) == 0:
        print("[{}s] {}".format(int(time.time() - t), i))
    design = np_designs[:, i, :]
    onsets = design[0]
    trial_idx = design[1]

    new_onsets = []
    trial_type = []
    files = []
    durations =[]
    itis = []
    trial_grp = []
    for j, idx in enumerate(trial_idx):
        # onset
        new_onsets.append(onsets[j])
        files.append(stim_db['File'][idx])
        durations.append(stim_db['Duration'][idx])
        itis.append(0.0)
        trial_type.append(stim_db['Condition'][idx])
        trial_grp.append(spk_names[stim_db['File'][idx][:3]])

        # answer
        new_onsets.append(onsets[j] + stim_db['Duration'][idx])
        files.append("")
        durations.append(5.0)
        if j < 35:
            itis.append(onsets[j+1] - onsets[j] - 5.0 - stim_db['Duration'][idx])
        else:
            itis.append(np.mean(itis))
        trial_type.append("answer")
        trial_grp.append("answer")

    isi_max = 0
    for j in range(len(new_onsets)-1):
        if isi_max < (new_onsets[j+1] - new_onsets[j]):
            isi_max = new_onsets[j+1] - new_onsets[j]
    isi_maxs.append(isi_max)

    designs.append({'onset': new_onsets, 'trial_type': trial_type, 'files': files, 'duration':
        durations, 'ITI': itis, 'trial_group': trial_grp})

pickle.dump({"designs": designs, "isi_maxs": isi_maxs}, open(
    "/hpc/banco/bastien.c/data/optim/identification/final_ans/designs.pck", "wb"))

# design = pd.read_csv("/hpc/banco/bastien.c/data/optim/identification/final/{}.csv".format(i))
#
# iti_v = []
# for k in range(len(design['onset'])-1):
#     iti_v.append(design['onset'][k+1] - design['onset'][k] - design['duration'][k])
# iti_v.append(np.mean(iti_v))
# design['ITI'] = iti_v
# #design['ITI'] = design['onset'][1:] - design['onset'][:-1] - design['duration'][:-1]
# #design['ITI'].append(np.mean(design['ITI']))
#
# design.to_csv("/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/identification_task/"
#               "{}.csv".format(i), index=False)

# for i in i_v:
#     print(i)
#     design = pd.read_csv("/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms"
#                          "/identification_task/{}.csv".format(i))
#     design = pd.DataFrame(add_answer_event(design, 5.0))
#     design.to_csv("/hpc/banco/bastien.c/data/fake_bids/sourcedata/paradigms/identification_task/"
#                   "past/designs_with_answers/{}.csv".format(i), index=False)
