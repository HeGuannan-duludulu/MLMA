import os
import shutil
from typing import Dict

import numpy as np
import pandas as pd
import pywt
import torch


def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        best_file_name: str,
        last_file_name: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    torch.save(state_dict, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, best_file_name))
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, last_file_name))


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_csv(csv_path="data/Physionet/labels/training-a/REFERENCE_withSQI.csv") -> Dict[str, int]:
    df = pd.read_csv(csv_path, usecols=[0, 1], header=None)
    first_col_list = df.iloc[:, 0].tolist()
    second_col_list = df.iloc[:, 1].tolist()
    label_dict = {name: label for name, label in zip(first_col_list, second_col_list)}
    return label_dict


def save_train_data(mfccs_array, label_array, save_dir: str):
    np.save((save_dir + "mfccs.npy"), mfccs_array)
    np.save((save_dir + "label.npy"), label_array)


def segment_audio(audio: np.ndarray, sample_rate, each_seg_time=2) -> list:
    audio_len = len(audio)
    each_seg_len = each_seg_time * sample_rate
    seg_num = int(np.ceil(audio_len / each_seg_len))
    seg_audio = []
    for i in range(seg_num):
        start = i * each_seg_len
        end = min((i + 1) * each_seg_len, audio_len)
        seg_audio.append(audio[start:end])
    return seg_audio


def calc_accuracy(feature_extractor, classifier, data_loader):
    feature_extractor.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.unsqueeze(1).float().cuda()
            targets = targets.long().cuda()
            features = feature_extractor(inputs)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


def calc_metrics(feature_extractor, classifier, data_loader):
    feature_extractor.eval()
    classifier.eval()
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.unsqueeze(1).float().cuda()
            targets = targets.long().cuda()
            features = feature_extractor(inputs)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            TP += ((predicted == 1) & (targets == 1)).sum().item()
            FP += ((predicted == 1) & (targets == 0)).sum().item()
            TN += ((predicted == 0) & (targets == 0)).sum().item()
            FN += ((predicted == 0) & (targets == 1)).sum().item()

    Se = TP / (TP + FN) if (TP + FN) != 0 else 0
    Sp = TN / (TN + FP) if (TN + FP) != 0 else 0

    MACC = (Se + Sp) / 2
    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    return Se, Sp, ACC, MACC


def dwt_denoise2(audio: np.ndarray, level) -> np.ndarray:
    coeffs = pywt.wavedec(audio, 'db1', level=level)
    cA2 = coeffs[0]
    zero_coeffs = [np.zeros_like(c) for c in coeffs[1:]]
    reconstructed_audio = pywt.waverec([cA2] + zero_coeffs, 'db1')
    return reconstructed_audio


def dwt_denoise(audio: np.ndarray, level=2) -> np.ndarray:
    coeffs = pywt.wavedec(audio, 'coif1', level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(audio)))

    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]

    print("coefficients:", len(denoised_coeffs[0]), len(denoised_coeffs[1]), len(denoised_coeffs[2]))
    reconstructed_audio = pywt.waverec(denoised_coeffs, 'coif1')

    return reconstructed_audio


if __name__ == "__main__":
    csv_path = "data/training-a/REFERENCE-SQI.csv"
    df = read_csv(csv_path)
    print(df)
