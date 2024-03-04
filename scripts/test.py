import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa


def load_data(normal_path, abbnormal_path):
    # compute the mfccs of normal audios and abnormal audios using librosa
    normal_mfccs = []
    abnormal_mfccs = []

    audio, sr = librosa.load(normal_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    normal_mfccs.append(mfccs)
    # calculate the first order difference
    mfccs_delta = librosa.feature.delta(mfccs)
    normal_mfccs.append(mfccs_delta)
    # calculate the second order difference
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    normal_mfccs.append(mfccs_delta2)


    audio, sr = librosa.load(abbnormal_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    abnormal_mfccs.append(mfccs)
    # calculate the first order difference
    mfccs_delta = librosa.feature.delta(mfccs)
    abnormal_mfccs.append(mfccs_delta)
    # calculate the second order difference
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    abnormal_mfccs.append(mfccs_delta2)

    return normal_mfccs, abnormal_mfccs
def show_mfccs_side_by_side(normal_mfccs, abnormal_mfccs):
    start = 680
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows for MFCCs, delta, and delta2; 2 columns for normal and abnormal
    for i in range(3):
        # Normal MFCCs and their deltas
        axes[i, 0].imshow(normal_mfccs[i][:, start:start+200], aspect='auto', origin='lower', cmap='viridis')
        titles = ['MFCCs', 'MFCCs Delta', 'MFCCs Delta2']
        axes[i, 0].set_title(f'Normal {titles[i]}', fontsize=16)

        # Abnormal MFCCs and their deltas
        axes[i, 1].imshow(abnormal_mfccs[i][:, start:start+200], aspect='auto', origin='lower', cmap='viridis')
        axes[i, 1].set_title(f'Abnormal {titles[i]}', fontsize=16)

    plt.tight_layout()
    save_path = "../images/mfccs_side_by_side.svg"
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    normal_path = "../data/training-e/e00001.wav"
    abnormal_path = "../data/training-e/e00020.wav"

    normal_mfccs, abnormal_mfccs = load_data(normal_path, abnormal_path)
    show_mfccs_side_by_side(normal_mfccs, abnormal_mfccs)