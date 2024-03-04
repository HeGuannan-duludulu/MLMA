
"""Compre the Mfccs with the normal and abnormal audios."""

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


def show_mfccs(normal_mfccs, abnormal_mfccs):
    # show the mfccs of normal audios and abnormal audios
    order = 2

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.title("Normal Audio")
    plt.imshow(normal_mfccs[order][:, :200], aspect='auto', origin='lower', cmap='viridis')

    plt.subplot(2, 1, 2)
    plt.title("Abnormal Audio")
    plt.imshow(abnormal_mfccs[order][:, :200], aspect='auto', origin='lower', cmap='viridis')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    normal_path = "../data/training-e/e00001.wav"
    abnormal_path = "../data/training-e/e00020.wav"

    normal_mfccs, abnormal_mfccs = load_data(normal_path, abnormal_path)
    show_mfccs(normal_mfccs, abnormal_mfccs)
