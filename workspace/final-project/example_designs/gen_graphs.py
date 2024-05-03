import re
import os
import matplotlib.pyplot as plt
import numpy as np

def extract_numbers_from_lines(file_path):
     with open(file_path, 'r') as file:
        for line in file:
            if "Utilization:" in line:
                number_match = re.search(r'\d+\.\d+', line)
                if number_match:
                    return float(number_match.group())

def extract_layer_number(filename):
    return int(re.search(r'r(\d+)', filename).group(1))

def get_data():
    data_dir1 = []
    data_dir2 = []
    file_dir1 = 'example_designs/eyeriss_like/outputs_DeepLabv3/'
    file_dir2 = 'example_designs/eyeriss_like/outputs_UndilatedDeepLabv3/'

    dir1 = os.listdir(file_dir1)
    dir2 = os.listdir(file_dir2)

    file_ls1 = sorted(dir1, key=extract_layer_number)
    file_ls2 = sorted(dir2, key=extract_layer_number)

    for file in file_ls1:
        file += '/timeloop-mapper.stats.txt'
        path = os.path.join(file_dir1, file)
        data_dir1.append(extract_numbers_from_lines(path))

    for file in file_ls2:
        file += '/timeloop-mapper.stats.txt'
        path = os.path.join(file_dir2, file)
        data_dir2.append(extract_numbers_from_lines(path))

    print(data_dir1, data_dir2)
    return data_dir1, data_dir2

def graph_data(data_dir1, data_dir2):
    x_values = np.array(range(len(data_dir1)))
    plt.plot(x_values, np.array(data_dir1), label="DeepLabv3")
    plt.plot(x_values, np.array(data_dir2), label="Undilated DeepLabv3")

    plt.xlabel('Layers')
    plt.ylabel('Metric')
    plt.title('DeepLabv3 v. Undilated DeepLabv3')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data_dir1, data_dir2 = get_data()
    graph_data(data_dir1, data_dir2)