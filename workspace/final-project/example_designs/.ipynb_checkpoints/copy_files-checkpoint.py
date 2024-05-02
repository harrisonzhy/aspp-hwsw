import re
import os
import shutil

def extract_layer_number(filename):
    return int(re.search(r'layer(\d+)', filename).group(1))

def file_iter():
    base_name = 'DeepLabv3_layer'
    file_dst = 'layer_shapes/CONV/DeepLabv3'
    file_dir1 = 'layer_shapes/CONV/resnet50'
    file_dir2 = 'layer_shapes/CONV/assp'

    file_ls1 = sorted(os.listdir(file_dir1), key=extract_layer_number)
    file_ls2 = sorted(os.listdir(file_dir2), key=extract_layer_number)

    ct = 0
    for filename in file_ls1:
        src = os.path.join(file_dir1, filename)
        new_filename = f"{base_name}{ct+1}.yaml"
        dst = os.path.join(file_dst, new_filename)
        shutil.copy(src, dst)
        ct += 1

    for filename in file_ls2:
        src = os.path.join(file_dir2, filename)
        new_filename = f"{base_name}{ct+1}.yaml"
        dst = os.path.join(file_dst, new_filename)
        shutil.copy(src, dst)
        ct += 1

if __name__ == "__main__":
    file_iter()