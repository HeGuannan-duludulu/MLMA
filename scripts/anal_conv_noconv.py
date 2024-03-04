import matplotlib.pyplot as plt
import re


def extract_data_from_logs(log_lines):
    data = {'epoch': [], 'training_loss': [], 'validation_loss': [], 'sensitivity': [], 'specificity': [], 'accuracy': [], 'macc': []}
    for line in log_lines:
        if 'Training Loss' in line:
            parts = line.split(',')
            data['epoch'].append(int(re.search(r'Epoch (\d+)/', parts[0]).group(1)))
            data['training_loss'].append(float(re.search(r'Training Loss: ([0-9.]+)', parts[1]).group(1)))
            data['validation_loss'].append(float(re.search(r'Validation Loss: ([0-9.]+)', parts[2]).group(1)))
            data['sensitivity'].append(float(re.search(r'Sensitivity: ([0-9.]+)%', parts[3]).group(1)))
            data['specificity'].append(float(re.search(r'Specificity: ([0-9.]+)%', parts[4]).group(1)))
            data['accuracy'].append(float(re.search(r'Accuracy: ([0-9.]+)%', parts[5]).group(1)))
            data['macc'].append(float(re.search(r'MACC: ([0-9.]+)%', parts[6]).group(1)))
    return data


with open('../log/training_withCONV.log', 'r') as file:
    log_lines_with_conv = file.readlines()

with open('../log/training_withoutdwt.log', 'r') as file:
    log_lines_without_conv = file.readlines()


data_with_conv = extract_data_from_logs(log_lines_with_conv)
data_without_conv = extract_data_from_logs(log_lines_without_conv)


plt.figure(figsize=(12, 6))
plt.plot(data_with_conv['epoch'], data_with_conv['training_loss'], label='Training Loss with Conv')
plt.plot(data_with_conv['epoch'], data_with_conv['validation_loss'], label='Validation Loss with Conv')
plt.plot(data_without_conv['epoch'], data_without_conv['training_loss'], label='Training Loss without Conv')
plt.plot(data_without_conv['epoch'], data_without_conv['validation_loss'], label='Validation Loss without Conv')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Comparison')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(data_with_conv['epoch'], data_with_conv['accuracy'], label='Accuracy with Conv')
plt.plot(data_without_conv['epoch'], data_without_conv['accuracy'], label='Accuracy without Conv')
plt.xlabel('Epoch')
plt.ylabel('Percentage')
plt.title('Accuracy Comparison')
plt.legend()
plt.show()
