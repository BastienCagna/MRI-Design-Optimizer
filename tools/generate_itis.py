#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

import matplotlib.pyplot as plt
import seaborn as sns


def generate_iti_vector(nbr, iti_min, sigma):
    """Generate ITIs based on a half Laplace distribution.

    :param nbr: Number of ITI.
    :param iti_min: Min value of ITI.
    :param sigma: Sigma of the Laplace distribution.
    :return: A vector of nbr ITI (in seconds).
    """
    iti_v = []
    i = 0
    while i < nbr:
        iti = np.random.laplace(iti_min, sigma)
        if iti > iti_min:
            iti_v.append(iti)
            i += 1

    return np.array(iti_v)


if __name__ == "__main__":
    iti_filename = sys.argv[1]

    resp = 'r'
    while resp == 'r':
        iti = generate_iti_vector(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))

        hist, h = np.histogram(iti, bins=31)
        plt.hist(iti, bins=31)
        plt.title("ITI distribution\nclose the window to continue")
        plt.text(0.9*max(iti), 0.8*max(hist), "min: {:.3f}\navg:  {:.3f}\nmax: {:.3f}".format(min(iti), np.mean(iti),
                                                                                       max(iti)))
        plt.show()

        resp = ''
        while resp != 'r' and resp != 's' and resp != 'q':
            resp = input("What you whant to do?\t\t'r' to restart - 'q' to quit - 's' to save and quit"
                         "\n\tyour choice: ")
            print("\n\n")

    if resp == 's':
        np.save(iti_filename, iti)
        print("{} saved.".format(iti_filename))