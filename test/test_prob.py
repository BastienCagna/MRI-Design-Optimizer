import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute(Ntot, ratio):
    t = time.time()
    Na = int(np.round(Ntot / (ratio + 1)))
    Nb = Ntot - Na

    ratio = Nb/float(Na)

    Npos = np.power(2, Ntot)
    print("Generating {} sequences".format(Npos))
    base = np.array([-1, 1])
    sequences = np.zeros((Npos, Ntot), dtype=int)
    for i in range(1, Ntot+1):
        sequences[:, i-1] = np.tile(base.repeat(np.power(2, i-1)), np.power(2, Ntot-i))

    print("Filtering sequences")

    final_sequences = []
    for i in range(Npos):
        avgs = moving_average(sequences[i], n=3)
        abmax = max([abs(np.min(avgs)), np.max(avgs)])
        if abmax < 1.0:
            if np.sum(sequences[i]==-1)==Na and np.sum(sequences[i]==1)==Nb:
                final_sequences.append(sequences[i])
    sequences = np.array(final_sequences)
    print("{} good sequences\n".format(sequences.shape[0]))

    print("Counting transitions")
    stats = np.array([0, 0, 0, 0])
    for seq in sequences:
        for i in range(Ntot-1):
            if seq[i] == -1 and seq[i+1] == -1:
                stats[0] += 1
            elif seq[i] == -1 and seq[i+1] == 1:
                stats[1] += 1
            elif seq[i] == 1 and seq[i+1] == -1:
                stats[2] += 1
            elif seq[i] == 1 and seq[i+1] == 1:
                stats[3] += 1

    stats = np.reshape(stats, (2, 2))

    print("Na: {}\tNb: {}\tNtot: {}\t ratio: {}".format(Na, Nb, Ntot, ratio))
    print(stats)

    tmn = np.zeros((2, 2))
    tmn[0] = stats[0] / [stats[0,0] + stats[0,1], stats[0,0] + stats[0,1]]
    tmn[1] = stats[1] / [stats[1,0] + stats[1,1], stats[1,0] + stats[1,1]]
    print(tmn)

    duration = time.time() - t
    return Na, Nb, Ntot, ratio, Npos, sequences.shape[0], duration, tmn[0, 0], tmn[0, 1], tmn[1, 0], tmn[1, 1]


def compute2(Ntot, ratio):
    t = time.time()
    Na = int(np.round(Ntot / (ratio + 1)))
    Nb = Ntot - Na

    ratio = Nb / float(Na)

    kmax = (Ntot/2) - 1
    nbr_seqs_max = np.power(2, kmax+1) + 2 * np.power(3, kmax)
    print("Nbr seqs max: {}".format(nbr_seqs_max))

    N = 2
    sequences = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    counts = [[2, 0], [1, 1], [1, 1], [0, 2]]
    while N < Ntot:
        # Stop condition for odd Ntot
        if np.mod(Ntot, 2) == 1 and N == Ntot-1:
            new_sequences = []
            for i, seq in enumerate(sequences):
                # If last condition is -1 and last two conditions was not -1, set -1
                if counts[i][0] < Na and seq[-2:] != [-1, -1]:
                    new_sequences.append(sequences[i] + [-1])
                # Else do the check for 1
                elif counts[i][1] < Nb and seq[-2:] != [1, 1]:
                    new_sequences.append(sequences[i] + [1])
                # If the last condition can't be set, remove the sequence
                else:
                    break
            sequences = new_sequences
            break

        new_sequences = []
        new_counts = []
        for i, seq in enumerate(sequences):
            if seq[-2:] == [-1, -1]:
                if counts[i][0] < Na and counts[i][1] < Nb:
                    new_sequences.append(seq + [1, -1])
                    new_counts.append([counts[i][0] + 1, counts[i][1] + 1])

                if counts[i][1] < Nb-1:
                    new_sequences.append(seq + [1, 1])
                    new_counts.append([counts[i][0], counts[i][1] + 2])

            elif seq[-2:] == [-1, 1]:
                if counts[i][0] < Na-1:
                    new_sequences.append(seq + [-1, -1])
                    new_counts.append([counts[i][0] + 2, counts[i][1]])

                if counts[i][0] < Na and counts[i][1] < Nb:
                    new_sequences.append(seq + [-1, 1])
                    new_counts.append([counts[i][0] + 1, counts[i][1] + 1])

                if counts[i][0] < Na and counts[i][1] < Nb:
                    new_sequences.append(seq + [1, -1])
                    new_counts.append([counts[i][0] + 1, counts[i][1] + 1])

            elif seq[-2:] == [1, -1]:
                if counts[i][0] < Na and counts[i][1] < Nb:
                    new_sequences.append(seq + [-1, 1])
                    new_counts.append([counts[i][0] + 1, counts[i][1] + 1])

                if counts[i][0] < Na and counts[i][1] < Nb:
                    new_sequences.append(seq + [1, -1])
                    new_counts.append([counts[i][0] + 1, counts[i][1] + 1])

                if counts[i][1] < Nb-1:
                    new_sequences.append(seq + [1, 1])
                    new_counts.append([counts[i][0], counts[i][1] + 2])

            else:
                if counts[i][0] < Na-1:
                    new_sequences.append(seq + [-1, -1])
                    new_counts.append([counts[i][0] + 2, counts[i][1]])

                if counts[i][0] < Na and counts[i][1] < Nb:
                    new_sequences.append(seq + [-1, 1])
                    new_counts.append([counts[i][0] + 1, counts[i][1] + 1])

        counts = new_counts
        sequences = new_sequences
        N += 2
    sequences = np.array(sequences)

    stats = np.array([0, 0, 0, 0])
    for seq in sequences:
        for i in range(Ntot - 1):
            if seq[i] == -1 and seq[i + 1] == -1:
                stats[0] += 1
            elif seq[i] == -1 and seq[i + 1] == 1:
                stats[1] += 1
            elif seq[i] == 1 and seq[i + 1] == -1:
                stats[2] += 1
            elif seq[i] == 1 and seq[i + 1] == 1:
                stats[3] += 1
    stats = np.reshape(stats, (2, 2))

    print("Na: {}\tNb: {}\tNtot: {}\t ratio: {}".format(Na, Nb, Ntot, ratio))
    print(stats)

    tmn = np.zeros((2, 2))
    tmn[0] = stats[0] / [stats[0, 0] + stats[0, 1], stats[0, 0] + stats[0, 1]]
    tmn[1] = stats[1] / [stats[1, 0] + stats[1, 1], stats[1, 0] + stats[1, 1]]
    print(tmn)

    duration = time.time()-t
    print("Nbr sequences: {}\tduration:{}".format(sequences.shape[0], duration))
    return Na, Nb, Ntot, ratio, nbr_seqs_max, sequences.shape[0], duration, tmn[0, 0], tmn[0, 1], tmn[1, 0], tmn[1, 1]

# N_v = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# ratios = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

N_v = [20, 30, 40, 50, 55, 60, 65, 70]
ratios = [0.4]

data = []
for N in N_v:
    for r in ratios:
        print("\n================= {}, {} ==================".format(N, r))
        data.append(compute2(N, r))
data = np.array(data)

np.save("/hpc/banco/bastien.c/data/optim/proba_simu", data)

data = np.load("/hpc/banco/bastien.c/data/optim/proba_simu.npy")

plt.figure()
plt.subplot(2, 2, 1)
plt.scatter(data[:, 2], data[:, 6])
plt.title("Duration vs. Ntot")
plt.subplot(2, 2, 2)
plt.scatter(data[:, 2], data[:, 5])
plt.title("Ngoods vs. Ntot")
plt.subplot(2, 2, 3)
plt.scatter(data[:, 2], data[:, 3])
plt.title("Tested points")
plt.xlabel("Ntot")
plt.ylabel("ratio")
plt.subplot(2, 2, 4)
plt.scatter(data[:, 3], data[:, 5])
plt.title("Ngoods vs. ratio")


fig = plt.figure()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
colors = sns.color_palette(flatui)

plt.subplot(1, 2, 1)
plt.scatter(data[:, 3], data[:, 7], c=colors[0])
plt.scatter(data[:, 3], data[:, 8], c=colors[1])
plt.scatter(data[:, 3], data[:, 9], c=colors[2])
plt.scatter(data[:, 3], data[:, 10], c=colors[3])
plt.title("Transition probabilities vs. ratio")
plt.xlabel("ratio")
plt.ylabel("Probabilities")
plt.legend(['AA', 'AB', 'BA', 'BB'])

plt.subplot(1, 2, 2)
plt.scatter(data[:, 2], data[:, 7], c=colors[0], s=60)
plt.scatter(data[:, 2], data[:, 8], c=colors[1], marker='s', s=50)
plt.scatter(data[:, 2], data[:, 9], c=colors[2], marker='o')
plt.scatter(data[:, 2], data[:, 10], c=colors[3], marker='^')
plt.xlabel("Ntot")
plt.ylabel("Probabilities")
plt.title("Transition probabilities vs. Ntot")
plt.legend(['AA', 'AB', 'BA', 'BB'])

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(data[:, 3], data[:, 5]/data[:, 4])
plt.xlabel("ratio")
plt.ylabel("Ngoods / Npos")
plt.subplot(1, 2, 2)
plt.scatter(data[:, 2], data[:, 5]/data[:, 4])
plt.xlabel("Ntot")
plt.ylabel("Ngoods / Npos")


plt.show()


