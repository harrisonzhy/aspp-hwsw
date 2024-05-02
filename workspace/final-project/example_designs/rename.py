import os

replacements = [("data-spaces", "data_spaces"), ("read-write", "read_write")]

def has_keyword(filename):
    keywords = ['conv', 'downsample']
    for keyword in keywords:
        if keyword in filename:
            return True
    return False

def get_layer_number(s):
    return int(s.split('[')[1].split(']')[0].split('layer')[1].split('_')[0])

def replace(file_path, old_text, new_text):
    with open(file_path, 'r') as file:
        file_content = file.read()
    modified_content = file_content.replace(old_text, new_text)
    with open(file_path, 'w') as file:
        file.write(modified_content)

def edit_files():
    folder_path = 'layer_shapes/CONV/assp'
    base_name = "assp_layer"

    file_list = os.listdir(folder_path)
    file_list = sorted(file_list, key=get_layer_number)

    for i, filename in enumerate(file_list):
        new_filename = f"{base_name}{i+1}.yaml"
        
        old_file_path = os.path.join(folder_path, filename)
        for old_text, new_text in replacements:
            replace(old_file_path, old_text, new_text)
        
        new_file_path = os.path.join(folder_path, new_filename)

        if has_keyword(old_file_path):
            os.rename(old_file_path, new_file_path)
        else:
            os.remove(old_file_path)

if __name__ == "__main__":
    edit_files()
    