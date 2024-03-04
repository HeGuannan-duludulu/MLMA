import numpy as np
import os
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def load_data(data_dir):
    features = []
    labels = []

    for file in os.listdir(data_dir):
        if file.endswith("_mfccs.npy"):
            mfccs = np.load(os.path.join(data_dir, file))
            label_file = file.replace("_mfccs.npy", "_label.npy")
            label = np.load(os.path.join(data_dir, label_file))

            for segment in range(mfccs.shape[0]):
                segment_features = mfccs[segment, :, :]

                avg_features = np.mean(segment_features, axis=0)
                features.append(avg_features)
                labels.append(label[segment])

    features = np.vstack(features)
    labels = np.array(labels)
    return features, labels


def train_predict_rf2(data_dir, test_size=0.2, random_state=42, n_estimators=5):
    features, labels = load_data(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    # y_pred_train = clf.predict(X_train)
    y_probs_test = clf.predict_proba(X_test)[:, 1]

    return y_probs_test, y_pred_test, y_test


def train_predict_rf(train_loader, valid_loader, n_estimators=5, random_state=42):
    X_train, y_train = extract_data(train_loader)
    X_test, y_test = extract_data(valid_loader)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    return y_pred_train, y_pred_test, y_train, y_test


def extract_data(loader):
    features = []
    labels = []
    for data, label in loader:
        flat_data = data.view(data.size(0), -1).numpy()
        features.extend(flat_data)
        labels.extend(label.numpy())
    return np.array(features), np.array(labels)


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

    return sensitivity, specificity, accuracy


def plot_roc_curve(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

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


def visualize_confusion_matrix_with_labels_enhanced(y_true, y_pred):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    tn, fp, fn, tp = tn / total, fp / total, fn / total, tp / total
    tn, fp, fn, tp = round(tn, 3), round(fp, 3), round(fn, 3), round(tp, 3)

    labels = np.array([['TN: ' + str(tn), 'FP: ' + str(fp)],
                       ['FN: ' + str(fn), 'TP: ' + str(tp)]])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=labels, fmt='', cmap='YlOrBr', linewidths=0.5, linecolor='black', cbar=False,
                annot_kws={"size": 16})
    plt.title('Random Forest', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    save_dir = "./images/" + "random_forest_matrix.png"
    plt.savefig(save_dir, dpi=300)
    plt.show()


if __name__ == "__main__":
    data_dir = config.train_data_dwt_save_dir
    y_probs_test, y_pred_test, y_test = train_predict_rf2(data_dir)
    plot_roc_curve(y_test, y_probs_test)

    print(classification_report(y_test, y_pred_test))

    sensitivity, specificity, accuracy = calculate_metrics(y_test, y_pred_test)
    print(f"Sensitivity: {sensitivity * 100:.2f}%")
    print(f"Specificity: {specificity * 100:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"MACC: {(sensitivity + specificity) / 2 * 100:.2f}%")
