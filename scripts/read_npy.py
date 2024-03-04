
"""Read .npy file"""

import numpy as np


mfccs = np.load("../data/feature_train_data/a0001_mfccs.npy")
label = np.load("../data/feature_train_data/a0011_label.npy")

print("mfccs.shape", mfccs.shape)
print("label.shape", label.shape)


import matplotlib.pyplot as plt

mfccs = np.load("../data/feature_train_data/a0001_mfccs.npy")
label = np.load("../data/feature_train_data/a0011_label.npy")

print("mfccs.shape", mfccs.shape)
print("label.shape", label.shape)

reshaped_mfccs = mfccs[0]
plt.figure(figsize=(15, 3))
# plot mfcc

plt.xlabel("Time(s)")
plt.ylabel("MFCCs")
plt.tight_layout()
plt.imshow(reshaped_mfccs.T[:13, :], aspect='auto', origin='lower', cmap='viridis')





# save svg image
plt.savefig("../images/mfcc.png", dpi=300)
plt.show()