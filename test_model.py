import torch
import torch.nn as nn

import config
from dataset import get_dataloader
from model import resnet_extractor, ImprovedABNClassifier
from utils import calc_accuracy

import matplotlib.pyplot as plt
import seaborn as sns
from svm import train_predict_svm2
from sklearn.metrics import roc_curve, auc
from random_forest import train_predict_rf2

def load_model(model_fe_path, model_abn_path) -> (nn.Module, nn.Module):
    model_fe = resnet_extractor(input_channel=1, num_layers=config.num_blocks, dropout=config.dropout).cuda()
    model_abn = ImprovedABNClassifier(model_fe.out_features, 2, dropout=config.abn_dropout).cuda()

    model_fe.load_state_dict(torch.load(model_fe_path))
    model_abn.load_state_dict(torch.load(model_abn_path))

    return model_fe, model_abn


from sklearn.metrics import confusion_matrix


def calculate_confusion_matrix(model_fe, model_abn, data_loader):
    all_preds = []
    all_labels = []
    model_fe.eval()
    model_abn.eval()
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.unsqueeze(1)

            inputs = inputs.cuda().float()
            labels = labels.cuda()

            outputs = model_abn(model_fe(inputs))
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)


def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
"""
if __name__ == '__main__':
    fe_model_path = "./data/BEST_resnet_model.pth"
    abn_model_path = "./data/BEST_abn_model.pth"
    model_fe, model_abn = load_model(fe_model_path, abn_model_path)
    _, test_loader = get_dataloader(config.train_data_dwt_save_dir)

    accuracy = calc_accuracy(model_fe, model_abn, test_loader)
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%')
    confusion_mat = calculate_confusion_matrix(model_fe, model_abn, test_loader)
    print(f'Confusion Matrix:\n{confusion_mat}')
    plot_confusion_matrix(confusion_mat)
    """


def calculate_roc_curve(model_fe, model_abn, data_loader):
    all_probs = []
    all_labels = []
    model_fe.eval()
    model_abn.eval()
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.unsqueeze(1)

            inputs = inputs.cuda().float()
            labels = labels.cuda()

            outputs = model_abn(model_fe(inputs))
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    fe_model_path = "./data/BEST_resnet_model.pth"
    abn_model_path = "./data/BEST_abn_model.pth"
    model_fe, model_abn = load_model(fe_model_path, abn_model_path)
    _, test_loader = get_dataloader(config.train_data_dwt_save_dir)
    fpr_res, tpr_res, roc_auc_res = calculate_roc_curve(model_fe, model_abn, test_loader)
    #plot_roc_curve(fpr, tpr, roc_auc)



    data_dir = config.train_data_dwt_save_dir

    y_probs_test, y_pred_test, y_test = train_predict_rf2(data_dir)
    y_probs_test_svm, y_pred_test_svm, y_test_svm = train_predict_svm2(data_dir)

    fpr, tpr, _ = roc_curve(y_test, y_probs_test)
    roc_auc = auc(fpr, tpr)

    fpr_svm, tpr_svm, _ = roc_curve(y_test_svm, y_probs_test_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)

    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='RF ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr_svm, tpr_svm, color='green', lw=2, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)
    #plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.plot(fpr_res, tpr_res, color='blue', lw=2, label='Our Model ROC curve (area = %0.2f)' % roc_auc_res)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")

    save_dir = "./images/ROC_Curve_Comparison.svg"
    plt.savefig(save_dir, dpi=300)
    plt.show()