import matplotlib.pyplot as plt
import numpy as np
import re

plt.style.use('ggplot')

log_file_paths = [
    "../log/training_withCONV.log",
    '../log/training_withoutCONV.log',
    '../log/training_withoutdwt.log',
]

max_metrics_all_methods = {}
model_name = ['CONV(ours)', 'withoutCONV', 'withoutDWT']

def parse_log_file(log_file_path):
    metrics_data = {
        'MACC': [],
        'Sensitivity': [],
        'Specificity': [],
        'Accuracy': []
    }

    with open(log_file_path, 'r') as file:
        for line in file:
            for metric in metrics_data.keys():
                match = re.search(rf'{metric}: (\d+\.\d+)', line)
                if match:
                    metrics_data[metric].append(float(match.group(1)))

    max_macc_index = metrics_data['MACC'].index(max(metrics_data['MACC']))
    max_metrics = {metric: values[max_macc_index] for metric, values in metrics_data.items()}

    return max_metrics



def plot_metrics(data):
    plt.figure(figsize=(12, 10))
    plt.title('Performance Metrics at Maximum MACC', fontsize=18)
    bar_width = 0.10
    spacing = 0.05
    colors = ['#E24A33',
              '#348ABD',
              '#988ED5',
              '#777777',
              '#FBC15E',
              '#8EBA42',
              '#FFB5B8']

    metrics_labels = list(data[model_name[0]].keys())
    index = np.arange(len(metrics_labels))

    for i, (method, metrics) in enumerate(data.items()):
        bars = plt.bar(index + i * (bar_width + spacing), metrics.values(), bar_width, color=colors[i], label=method)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), ha='center', va='bottom', fontsize=14)

    plt.xlabel('Metrics', fontsize=14)
    plt.xticks(index + bar_width, metrics_labels, fontsize=15)
    plt.ylabel('Value [%]', fontsize=14)
    plt.legend(fontsize=12, loc='lower center')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_path = '../images/metrics.svg'
    plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    for i, path in enumerate(log_file_paths):
        max_metrics = parse_log_file(path)
        max_metrics_all_methods[model_name[i]] = max_metrics

    max_metrics_all_methods['SVM'] = {
        'MACC': 80.97,
        'Sensitivity': 68.37,
        'Specificity': 93.57,
        'Accuracy': 87.58
    }

    max_metrics_all_methods['Random Forest'] = {
        'MACC': 84.46,
        'Sensitivity': 75.38,
        'Specificity': 93.54,
        'Accuracy': 89.22
    }

    plot_metrics(max_metrics_all_methods)


if __name__ == "__main__":
    main()
