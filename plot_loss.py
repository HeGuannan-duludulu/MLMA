import re
import matplotlib.pyplot as plt

dir_path = "training.log"

try:
    with open(dir_path, 'r') as f:
        lines = f.readlines()
        loss_list = [re.findall(r"Loss: (\d+\.\d+)", line)[0] for line in lines if "Loss" in line]
    plt.plot([float(value) for value in loss_list])
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print(f"File {dir_path} not found!")
except Exception as e:
    print(f"Error：{e}")
