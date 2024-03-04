"""Show wav from the wav file in the PCG dataset."""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import config

# read wav file
wav_file = os.path.join("../data/training-a/a0013.wav")
wav, sr = librosa.load(wav_file, sr=None)

# plot wav
plt.figure(figsize=(15, 3))
plt.axis('off')
plt.tight_layout()
plt.plot(wav[1800: 4500])
save_dir = "../images/" + "wav.svg"
# save svg image
plt.savefig(save_dir, dpi=300)


plt.show()
