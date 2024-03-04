
import os
import random

import torch

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
random.seed(0)

mode = "train"
epochs = 300

lr = 0.001
lr_scheduler_step_size = epochs // 4
lr_scheduler_gamma = 0.8

batch_size = 128
num_blocks = 3

data_path = "./data"

train_data_save_dir = "data/feature_train_data"
test_data_save_dir = "data/feature_test_data"

train_data_dwt_save_dir = "data/all_dwt"


results_dir = "./train_dir/results"
dropout = 0.1
abn_dropout = 0.4


valid_print_interval = 5


if mode == "train":
    train_model_save_dir = "./train_dir/models"
    if not os.path.isdir(train_model_save_dir):
        os.makedirs(train_model_save_dir)


if mode == "test":
    model_read_path = "./train_dir/models/g_epoch_100.pth.tar"