import os
import random
import cv2
from PIL import Image
from .files_and_folders import generate_folder, remove_style_name

sizes = [0.4, 0.55, 0.6, 0.65, 0.72, 0.78, 0.85, 0.9, 0.95, 1]
repeats = [1, 2, 2, 3, 2, 3, 2, 2, 1, 1]

def generate_crops(image_path, image_name, output_folder, prefix=""):
    if prefix != "" and not prefix.endswith('_'):
        prefix += '_'
    image_name = prefix + image_name
    with Image.open(image_path) as img:
        total_crops = []
        for threshold in range(7, 4, -1):
            crops = []
            for j in range(len(sizes)):
                g_size = int(img.height * sizes[j])
                for i in range(repeats[j]):
                    l1 = random.randint(0, img.width - g_size)
                    # l2 = random.randint(0, img.width - g_size)
                    # left = (0 + l1 + l2) // 3
                    t1 = random.randint(0, img.height - g_size)
                    # t2 = random.randint(0, img.height - g_size)
                    # t3 = random.randint(0, img.height - g_size)
                    # top = (t1 + t2 + t3) // 3
                    # cropped = img.crop((left, top, left + g_size, top + g_size))
                    cropped = img.crop((l1, t1, l1 + g_size, t1 + g_size))
                    crops.append(cropped)
            total_crops += crops
        # for i, crop in enumerate(total_crops):
        #     output_path = os.path.join(output_folder,
        #                                f"{image_name}_crop{i + 1}.png")
        #
        #     crop.save(output_path)
        # img.save(os.path.join(output_folder, f"{image_name}_1.png"))
        # img.save(os.path.join(output_folder, f"{image_name}_2.png"))


def preprocess_single_scene(root, output_folder="", unique_folder=False, prefix=""):
    if output_folder == "":
        output_folder = generate_folder(os.path.split(root)[1], "TempDatasets", unique_folder=unique_folder)
    if prefix != "":
        prefix += '_'
    image_files = os.listdir(root)

    for idx, image_name in enumerate(image_files):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        image_path = os.path.join(root, image_name)
        generate_crops(image_path, os.path.splitext(image_name)[0], output_folder, prefix=prefix)

    return output_folder

def preprocess_multiple_scenes(root, output_folder="", unique_folder=False, prefix=""):
    if output_folder == "":
        output_folder = generate_folder(os.path.split(root)[1], "TempDatasets", unique_folder=unique_folder)

    if prefix != "":
        prefix += '_'
    folders = os.listdir(root)

    for f in folders:
        folder_path = os.path.join(root, f)
        rsn = remove_style_name(f)
        print(rsn)
        preprocess_single_scene(folder_path, output_folder, prefix=p)


    return output_folder
