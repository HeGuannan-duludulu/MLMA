import matplotlib.pyplot as plt
import re

model_name = ['CONV(ours)', 'withoutCONV', 'withoutDWT']


def parse_log_file(log_file_path):
    epochs = []
    losses = []

    with open(log_file_path, 'r', encoding="utf=8") as file:
        for line in file:
            match = re.search(r'Epoch (\d+)/\d+, Loss: ([0-9.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)

    return epochs, losses


def enhanced_plot(data):
    plt.figure(figsize=(10, 6))

    line_styles = ['-', '-', '-']
    colors = ['b', 'g', 'r']

    for i, (epochs, losses) in enumerate(data):
        plt.plot(epochs, losses, label=model_name[i], linestyle=line_styles[i])

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Comparison', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)

    save_path = "../images/enhanced_loss_plot.svg"
    plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    log_file_paths = [
        "../log/training_withCONV.log",
        "../log/training_withoutCONV.log",
        "../log/training_withoutdwt.log",
    ]

    data = []
    for path in log_file_paths:
        epochs, losses = parse_log_file(path)
        data.append((epochs, losses))

    enhanced_plot(data)


if __name__ == "__main__":
    main()
