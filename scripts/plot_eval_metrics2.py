import matplotlib.pyplot as plt
import numpy as np
import re

log_file_paths = [
    "../log/training_withCONV.log",
    '../log/training_withoutCONV.log',
    '../log/training_withoutdwt.log',
]


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


# 绘制图表
def plot_metrics(data):
    plt.figure(figsize=(14, 8))
    plt.title('Performance Metrics at Maximum MACC for Each Method')
    bar_width = 0.2
    colors = ['b', 'g', 'r', 'c']
    index = np.arange(len(data))

    for i, (metric, values) in enumerate(data[list(data.keys())[0]].items()):
        method_values = [data[method][metric] for method in data]
        plt.bar(index + i * bar_width, method_values, bar_width, color=colors[i], label=metric)

    plt.xlabel('Methods')
    plt.xticks(index + bar_width / 2, data.keys())
    plt.ylabel('Value [%]')
    plt.legend(title="Metrics")
    plt.show()


def main():
    max_metrics_all_methods = {}
    for i, path in enumerate(log_file_paths):
        max_metrics = parse_log_file(path)
        max_metrics_all_methods[f'Method {i + 1}'] = max_metrics

    plot_metrics(max_metrics_all_methods)


if __name__ == "__main__":
    main()
