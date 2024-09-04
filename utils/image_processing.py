import os
import random
import json
import cv2
from PIL import Image
from .files_and_folders import generate_folder, remove_style_name, read_folders_from_json

sizes = [0.7, 0.73, 0.77, 0.85, 0.9, 0.95]
repeats = [2, 2, 3, 3, 2, 3]

calculate_sizes = [
    lambda img, size: (img.width * size, img.height * size),
    lambda img, size: (img.width * size, img.height * size),
    lambda img, size: (img.width * size, img.height * size),
    lambda img, size: (img.width * size, min(img.height, img.width * size)),
    lambda img, size: (min(img.height * size, img.width), img.height * size)
]
N = len(calculate_sizes)
def generate_crops(image_path, image_name, output_folder, prefix=""):
    if prefix != "" and not prefix.endswith('_'):
        prefix += '_'
    image_name = prefix + image_name
    with Image.open(image_path) as img:
        img = img.resize((960,540))
        total_crops = [(image_name, img)]
        crops = []
        for j in range(len(sizes)):
            rand_idx = random.randint(0, int(N-1))
            w_size, h_size = calculate_sizes[rand_idx](img, sizes[j])
            for i in range(repeats[j]):
                l1 = random.randint(0, int(img.width - w_size))
                t1 = random.randint(0, int(img.height - h_size))
                cropped = img.crop((l1, t1, l1 + w_size, t1 + h_size))
                crops.append((image_name, cropped))
        total_crops += crops

    return total_crops


def preprocess_single_scene(root, folder_base, test_details,  folder_name=None, output_folder="", max_crops=3000, unique_folder=False, prefix="", only_2_frames=True):
    if output_folder == "":
        output_folder = generate_folder(folder_name, folder_base=folder_base, root_path="TempDatasets", test_details=test_details, unique_folder=unique_folder)
    if prefix != "":
        prefix += '_'
    image_files = os.listdir(root)

    if len(image_files) < 3:
        only_2_frames = False
    idx = -1
    total_crops = []
    for image_name in image_files:
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        idx += 1
        if only_2_frames and "frame2" in image_name:
            continue
        image_path = os.path.join(root, image_name)
        total_crops += generate_crops(image_path, os.path.splitext(image_name)[0], output_folder, prefix=prefix)

    if max_crops is None:
        return total_crops
    if len(total_crops) > max_crops:
        total_crops = random.sample(total_crops, max_crops)
    for i, data in enumerate(total_crops):
        name, image = data
        image.save(os.path.join(output_folder, f"{name}_crop{i + 1}.png"))
    return output_folder

def preprocess_multiple_scenes(root, folder_base, test_details, output_folder="", max_crops=3000, unique_folder=False, prefix="", only_2_frames=True):
    if os.path.isfile(root):
        return preprocess_multiple_scenes_json(root, output_folder, unique_folder, prefix, only_2_frames)
    
    folders = os.listdir(root)
    if output_folder == "":
        output_folder = generate_folder(folder_base=folder_base, root_path="TempDatasets", test_details=test_details, unique_folder=unique_folder)
    # else:
    #     raise ValueError("Can't find this folder")

    if prefix != "":
        prefix += '_'

    total_crops = []
    for f in folders:
        folder_path = os.path.join(root, f)
        # rsn = remove_style_name(f)
        rsn = f
        p = prefix + rsn
        total_crops += preprocess_single_scene(folder_path, folder_base=folder_base, test_details=test_details, output_folder=output_folder, max_crops=None, prefix=p, only_2_frames=only_2_frames)

    if len(total_crops) > max_crops:
        total_crops = random.sample(total_crops, max_crops)
    for i, data in enumerate(total_crops):
        name, image = data
        image.save(os.path.join(output_folder, f"{name}_crop{i + 1}.png"))

    return output_folder

def preprocess_multiple_scenes_json(json_file, output_folder="", unique_folder=False, prefix="", only_2_frames=True):
    # folders = read_folders_from_json(root)
    if output_folder == "":
        x = os.path.splitext(os.path.split(json_file)[1])[0]
        output_folder = generate_folder(x, "TempDatasets", unique_folder=unique_folder)

    # else:
    #     raise ValueError("Can't find this folder")
    
    if prefix != "":
        prefix += '_'
    #
    # for f in folders:
    #     rsn = remove_style_name(f)
    #     p = prefix + rsn
    #     preprocess_single_scene(folder_path, output_folder, prefix=p)

    with open(json_file, 'r') as f:
        data = json.load(f)

    for root, folders in data.items():
        for folder in folders:
            rsn = remove_style_name(folder["folder"])
            p = prefix + rsn
            preprocess_single_scene(os.path.join(root, folder["folder"]), output_folder, prefix=p, only_2_frames=only_2_frames)
    
    return output_folder
