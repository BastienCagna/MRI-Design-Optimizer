import nibabel.gifti as ng
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import sys


file = sys.argv[1]

# Read gifti file
gii_img = ng.read(file)

data = gii_img.darrays[0].data

print("File: {}".format(file))

print("\nData shape:")
print(data.shape)

vmin = min(data)
vmax = max(data)
mean = np.mean(data)
var = np.var(data)

# Histogram
fig = plt.figure()
plt.hist(data, bins=100)
plt.text(4, 5000, "vmin: {:.02f}\nvmax: {:.02f}\n\navg: {:.02f}\nvar: {:.02f}".format(vmin, vmax,
                                                                                      mean, var))
plt.title(file)

plt.show()
