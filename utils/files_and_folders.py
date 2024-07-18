import os
from datetime import datetime


def get_next_test_number(root_path, current_date):
    test_dir = os.path.join(root_path, current_date)
    if not os.path.exists(test_dir):
        return 1

    existing_tests = [d for d in os.listdir(test_dir) if
                      d.startswith("test") and os.path.isdir(os.path.join(test_dir, d))]
    if not existing_tests:
        return 1

    max_test_num = max(int(d[4:]) for d in existing_tests)
    return max_test_num + 1

def generate_folder(folder_name=None, root_path="checkpoints/outputs/LoRAs", extension=None, unique_folder=False):
    # Get current date in mm-dd format
    current_date = datetime.now().strftime("%m-%d")
    # Get the next test number
    test_num = get_next_test_number(root_path, current_date)
    if not unique_folder and test_num > 1:
        test_num -= 1

    # Create the full path
    # full_path = os.path.join(root_path, current_date, f"test{test_num}", folder_name)
    if folder_name is not None:
        if extension:
            full_path = os.path.join(root_path, current_date, f"test{test_num}", extension, folder_name)
        else:
            full_path = os.path.join(root_path, current_date, f"test{test_num}", folder_name)
    else:
        if extension:
            full_path = os.path.join(root_path, current_date, f"test{test_num}", extension)
        else:
            full_path = os.path.join(root_path, current_date, f"test{test_num}")
    # Create the directory
    os.makedirs(full_path, exist_ok=True)

    print(f"Folder created: {full_path}")
    return full_path

def extract_style_name(file_name):
    if file_name.startswith("Disney"):
        return "Disney"
    elif file_name.startswith("Pixar"):
        return "Pixar"
    elif file_name.startswith("Japan") or file_name.startswith("Anime"):
        return "Anime"
    return "Animation"


def remove_style_name(file_name):
    style = extract_style_name(file_name)
    if file_name.startswith(style):  # for cases that style = "Animation" and the file's name doesn't start with "Animation"
        return file_name.replace((style + '_'), "")
    elif style == "Anime":
        return file_name.replace("Japan_", "")
    return file_name