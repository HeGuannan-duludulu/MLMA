import logging

import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloader
from utils import calc_accuracy, calc_metrics
from model import resnet_extractor, abn_classifier, ImprovedABNClassifier

logging.basicConfig(filename='./log/training_withCONV.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_seed(seed)


def validate(model_fe, model_abn, valid_loader, criterion):
    model_fe.eval()
    model_abn.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.unsqueeze(1)
            inputs = inputs.float().cuda()
            targets = targets.long().cuda()

            outputs = model_fe(inputs)
            cls = model_abn(outputs)
            loss = criterion(cls, targets)
            val_loss += loss.item()

    return val_loss / len(valid_loader)


train_loader, valid_loader = get_dataloader(config.train_data_dwt_save_dir)

model_fe = resnet_extractor(input_channel=1, num_layers=config.num_blocks, dropout=config.dropout).cuda()
# model_abn = abn_classifier(model_fe.out_features, 2, dropout=config.dropout).cuda()
model_abn = ImprovedABNClassifier(model_fe.out_features, 2, dropout=config.abn_dropout).cuda()

weights = torch.tensor([0.8, 1.3], dtype=torch.float32).cuda()
criterion = nn.CrossEntropyLoss(weight=weights)

print("build loss func success")
optimizer_fe = optim.Adam(model_fe.parameters(), lr=config.lr)
optimizer_abn = optim.Adam(model_abn.parameters(), lr=config.lr)
print("build optimizer success")

scheduler_fe = optim.lr_scheduler.StepLR(optimizer_fe, step_size=config.lr_scheduler_step_size,
                                         gamma=config.lr_scheduler_gamma)
scheduler_abn = optim.lr_scheduler.StepLR(optimizer_abn, step_size=config.lr_scheduler_step_size,
                                          gamma=config.lr_scheduler_gamma)

highest_macc = 0.0

for epoch in range(config.epochs):
    model_fe.train()
    model_abn.train()
    epoch_loss = 0

    for inputs, targets in train_loader:
        inputs = inputs.unsqueeze(1)
        inputs = inputs.float().cuda()
        targets = targets.long().cuda()

        optimizer_fe.zero_grad()
        optimizer_abn.zero_grad()

        outputs = model_fe(inputs)
        cls = model_abn(outputs)
        loss = criterion(cls, targets)

        loss.backward()
        optimizer_fe.step()
        optimizer_abn.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    logging.info(f"Epoch {epoch + 1}/{config.epochs}, Loss: {average_loss:.4f}")
    print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {average_loss:.4f}")

    scheduler_fe.step()
    scheduler_abn.step()

    if (epoch + 1) % 5 == 0:
        Se, Sp, ACC, MACC = calc_metrics(model_fe, model_abn, valid_loader)
        val_loss = validate(model_fe, model_abn, valid_loader, criterion)
        logging.info(
            f"Epoch {epoch + 1}/{config.epochs}, Training Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}, "
            f"Sensitivity: {Se * 100:.2f}%, Specificity: {Sp * 100:.2f}%, Accuracy: {ACC * 100:.2f}%, MACC: "
            f"{MACC * 100:.2f}%")
        print(
            f"Epoch {epoch + 1}/{config.epochs}, Training Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}, "
            f"Sensitivity: {Se * 100:.2f}%, Specificity: {Sp * 100:.2f}%, Accuracy: {ACC * 100:.2f}%, MACC: "
            f"{MACC * 100:.2f}%")

        if MACC > highest_macc:
            highest_macc = MACC
            torch.save(model_fe.state_dict(), './data/BEST_resnet_model.pth')
            torch.save(model_abn.state_dict(), './data/BEST_abn_model.pth')

    if (epoch + 1) % 50 == 0:
        torch.save(model_fe.state_dict(), './data/resnet_model.pth')
        torch.save(model_abn.state_dict(), './data/abn_model.pth')

Se, Sp, ACC, MACC = calc_metrics(model_fe, model_abn, valid_loader)
print(f'Test set metrics: Sensitivity: {Se * 100:.2f}%, Specificity: {Sp * 100:.2f}%, Accuracy: {ACC * 100:.2f}%')
