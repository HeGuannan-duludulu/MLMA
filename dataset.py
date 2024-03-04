import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import config



def load_dataset(data_folder):
    mfcc_files = glob.glob(os.path.join(data_folder, '*_mfccs.npy'))
    label_files = {os.path.basename(f).split('_')[0]: f for f in glob.glob(os.path.join(data_folder, '*_label.npy'))}

    data_segments = []

    for mfcc_file in mfcc_files:
        base_name = os.path.basename(mfcc_file).split('_')[0]
        corresponding_label_file = label_files.get(base_name)

        if corresponding_label_file:
            mfccs = np.load(mfcc_file)
            labels = np.load(corresponding_label_file)

            for segment, label in zip(mfccs, labels):
                data_segments.append((segment, label))

    data_positive = [seg for seg in data_segments if seg[1] == 1]
    data_negative = [seg for seg in data_segments if seg[1] == 0]

    return data_positive, data_negative


def split_balanced_dataset(data_positive, data_negative, test_size=0.2):
    test_pos_size = int(test_size * len(data_positive))
    test_neg_size = int(test_size * len(data_negative))
    test_pos_size = min(test_pos_size, len(data_positive))
    test_neg_size = min(test_neg_size, len(data_negative))
    print("test_pos_size", test_pos_size, "test_neg_size", test_neg_size)

    pos_train, pos_test = train_test_split(data_positive, test_size=test_pos_size, random_state=42)
    neg_train, neg_test = train_test_split(data_negative, test_size=test_neg_size, random_state=42)

    train_data = pos_train + neg_train
    test_data = pos_test + neg_test

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    return train_data, test_data


class HeartSoundDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mfcc_segment, label = self.data[idx]
        return mfcc_segment, int(label)


def get_dataloader(data_folder):
    data_positive, data_negative = load_dataset(data_folder)
    train_data, test_data = split_balanced_dataset(data_positive, data_negative)
    print("len(train_data)", len(train_data), "len(test_data)", len(test_data))
    print(type(train_data), type(test_data))
    train_dataset = HeartSoundDataset(train_data)
    valid_dataset = HeartSoundDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, valid_loader


if __name__ == "__main__":
    data_folder = config.train_data_save_dir

    train_loader, valid_loader = get_dataloader(data_folder)
    print("len(train_loader)", len(train_loader))
    print("len(valid_loader)", len(valid_loader))
