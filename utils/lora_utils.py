import os
import random
from datetime import datetime

import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import torch
import json


custom_cache_dir = "/cs/labs/danix/nadav_al/AnimeInterp/checkpoints/Blip"
os.environ['HF_HOME'] = custom_cache_dir
#
# Set up the BLIP image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=custom_cache_dir)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=custom_cache_dir, torch_dtype=torch.float16).to("cuda")

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

def generate_folder(folder_name=None, root_path="sd-model-fintuned-lora", extension=None, unique_folder=False):
    # breakpoint()
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


from enum import Enum
class Methods(Enum):
    RANDOM_CROP = 0
    RANDOM_SIZE = 1
    GRID = 2
    JITTER = 3
    GRID_RANDOM = 4
    JITTER_RANDOM = 5

def random_size_crop(img, size, target_size, min_crop_ratio, max_crop_ratio, num_crops):
    width, height = img.size
    crop_width, crop_height = size
    crops = []
    for _ in range(num_crops):
        crop_ratio = random.randint(int(min_crop_ratio*10), int(max_crop_ratio*10)+1)
        aug_crop_width = int(crop_width * (crop_ratio/10))
        aug_crop_height = int(crop_height * (crop_ratio/10))
        if aug_crop_height > height:
            aug_crop_height = crop_height
        if aug_crop_width > width:
            aug_crop_width = crop_width

        left = random.randint(0, width - aug_crop_width)
        top = random.randint(0, height - aug_crop_height)
        left = max(left, 0)
        top = max(top, 0)
        cropped = img.crop((left, top, left+aug_crop_width, top+aug_crop_height))
        resized = cropped.resize((target_size, target_size), Image.LANCZOS)
        crops.append(resized)
    return crops

def jitter_random_crop(img, size, target_size, min_crop_ratio=0.6, max_crop_ratio=1.3, overlap=0.36, jitter_range=0.25):
    width, height = img.size
    crop_width, crop_height = size
    step_x = int(crop_width * (1 - overlap))
    step_y = int(crop_height * (1 - overlap))
    crops = []
    # margin = int(height * jitter_range / 3)
    margin = 0
    for y in range(margin, height - (crop_height + margin - 1), step_y):
        for x in range(margin, width - (crop_width + margin - 1), step_x):
            # print(f"x={x}, y={y}")
            # Apply random jitter within the step range
            jitter_x = int(random.uniform(-jitter_range, jitter_range) * step_x)
            jitter_y = int(random.uniform(-jitter_range, jitter_range) * step_y)

            # Adjust for potential boundary violations due to jitter
            left = max(0, min(x + jitter_x, width - crop_width))
            top = max(0, min(y + jitter_y, height - crop_height))

            # Randomly determine crop size within the specified ratio range
            crop_ratio = random.randint(int(min_crop_ratio*10), int(max_crop_ratio*10))
            aug_crop_width = int(crop_width * crop_ratio/10)
            aug_crop_height = int(crop_height * crop_ratio/10)

            aug_left = max(left - abs(aug_crop_width-crop_width)/2, 0)
            aug_top = max(top - abs(aug_crop_height-crop_height)/2, 0)

            # Ensure crop stays within image boundaries
            aug_crop_width = aug_crop_width if (aug_left+aug_crop_width) <= width else crop_width
            aug_crop_height = aug_crop_height if (aug_left + aug_crop_height) <= height else crop_height

            # Extract the random crop with jitter
            cropped = img.crop((aug_left, aug_top, aug_left + aug_crop_width, aug_top + aug_crop_height))
            # print(f"left={aug_left}, top={aug_top}, right={aug_left+aug_crop_width}, bottom={aug_top+aug_crop_height}")

            # Resize the crop to the final desired size
            resized_crop = cropped.resize((target_size, target_size), Image.LANCZOS)
            crops.append(resized_crop)
    return crops

def calculate_color_std(image):
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = img_array.astype(np.float32) / 255.0
    return np.mean(img_array)




def filter_crops(crops, img_mean, threshold):
    f_crops = []
    d_crops = []
    acc = []
    dec = []
    for crop in crops:
        std = abs(calculate_color_std(crop) - img_mean)
        if std <= threshold:
            f_crops.append(crop)
            acc.append(std)
        else:
            d_crops.append(crop)
            dec.append(std)
    return f_crops, d_crops

sizes = [0.4, 0.55, 0.6, 0.65, 0.72, 0.78, 0.85, 0.9, 0.95, 1]
repeats = [2, 2, 2, 3, 2, 3, 2, 2, 1, 1]

def generate_crops_from_image(img, repeat=1, num_crops=0, methods=None):
    total_crops = []
    sampled = False
    for klkl in range(repeat+1):
        print("entered ", klkl)
        for j in range(len(sizes)):
            crops = []
            w_crop_ratio = random.randint(int(img.height * sizes[j]), int(img.width * sizes[j]))
            h_crop_ratio = random.randint(int(img.height * sizes[j]), int(img.width * sizes[j]))
            h_crop_ratio = min(h_crop_ratio, img.height)

            for i in range(repeats[j]):
                l1 = random.randint(0, img.width - w_crop_ratio)
                # l2 = random.randint(0, img.width - w_crop_ratio)
                # left = (0 + l1 + l2) // 3
                left = l1
                t1 = random.randint(0, img.height - h_crop_ratio)
                # t2 = random.randint(0, img.height - h_crop_ratio)
                # t3 = random.randint(0, img.height - h_crop_ratio)
                # top = (t1 + t2 + t3) // 3
                top = t1
                cropped = img.crop((left, top, left + w_crop_ratio, top + h_crop_ratio))
                crops.append(cropped)
            if num_crops > 0 and len(crops) > num_crops:
                sampled = True
                crops = random.sample(crops, num_crops)
            total_crops += crops
        if num_crops > 0 and not sampled:
            total_crops = random.sample(total_crops, num_crops)
    return total_crops


def preprocess_single_folder(images_folder, num_crops=0, max_crops=200, methods=None, unique_folder=False,
                             save_folder=True):
    output_folder = ""
    if save_folder:
        output_folder = generate_folder(os.path.split(images_folder)[1], "TempDatasets", unique_folder=unique_folder)
    if methods is None:
        methods = [Methods.JITTER_RANDOM]
    image_files = os.listdir(images_folder)
    image_files.sort()
    all_crops = []
    total_amount = 0
    for idx, image_name in enumerate(image_files):
        if idx > 3 or idx == 1 or not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        image_path = os.path.join(images_folder, image_name)
        image_name = os.path.splitext(image_name)[0]
        with Image.open(image_path) as img:
            crops = generate_crops_from_image(img, num_crops)
            print(len(crops))
            all_crops.append([image_name, crops])
            total_amount += len(crops)


    do_sample = False
    sample_rate = max_crops
    if total_amount > max_crops:
        do_sample = True
        sample_rate = max_crops // len(all_crops)

    for i, crop_data in enumerate(all_crops):
        image_name, crop_lst = crop_data
        if do_sample:
            crop_lst = random.sample(crop_lst, sample_rate)

        if save_folder:
            for crop in crop_lst:
                output_path = os.path.join(output_folder, f"{image_name}_crop{i + 1}.png")
                crop.save(output_path)
            img = Image.open(os.path.join(images_folder, f"{image_name}.png"))
            img.save(os.path.join(output_folder, f"{image_name}_1.png"))
            img.save(os.path.join(output_folder, f"{image_name}_2.png"))
        else:
            all_crops[i][1] = crop_lst
    return output_folder if save_folder else all_crops


def preprocess_multiple_folders(root, num_crops=0, max_crops=200, methods=None, unique_folder=False, save_folder=True):
    root_name = os.path.split(root)[1]
    output_folder = ""
    if save_folder:
        output_folder = generate_folder(root_name, "TempDatasets", unique_folder=unique_folder)
    if methods is None:
        methods = [Methods.JITTER_RANDOM]
    folders = os.listdir(root)
    folders.sort()
    all_crops = []
    total_amount = 0
    for folder in folders:
        folder_crops = preprocess_single_folder(os.path.join(root, folder), num_crops=num_crops, methods=methods, unique_folder=unique_folder, save_folder=False)
        for image_name, crop_lst in folder_crops:
            total_amount += len(crop_lst)
            for i, crop in enumerate(crop_lst):
                all_crops.append([folder, image_name, i, crop])

    if len(all_crops) > max_crops:
        all_crops = random.sample(all_crops, max_crops)

    if save_folder:
        for crop_data in all_crops:
            folder, image_name, i, crop = crop_data
            output_path = os.path.join(output_folder, f"{folder}_{image_name}_crop{i}.png")
            crop.save(output_path)

    return output_folder if save_folder else all_crops



UNIQUE_TOKEN = 'dinep'
KEYWORDS = {
    UNIQUE_TOKEN,
    'best quality',
    'high quality',
    'cel shading',
    'colorfull',
    'drawing',
    'Cartoon',
    'Classic Cartoon',
    'Hard Shadows',
    'Vibrant Colors',
    'Flat Color',
    'Frame-by-Frame',
    # 'Lineart',
    # 'clear lines',
    'Motion',
    'Extreme Poses',
    'Hand-Drawn',
    '2D',
}
STYLE_TOKENS = {
    'Disney': ['Disney-Renaissance', f'{UNIQUE_TOKEN} Disney', 'Disney', 'Animation'],
    'Anime': ['Anime', 'StudioGhibli', f'{UNIQUE_TOKEN} Anime', f'Ghibli {UNIQUE_TOKEN}'],
    'Pixar': ['Pixar', f'{UNIQUE_TOKEN} Pixar', '3D', f'3D {UNIQUE_TOKEN}'],
    'Animation': ['Animation', f'{UNIQUE_TOKEN} Animation', f'{UNIQUE_TOKEN} Drawing']
}

def generate_caption(image, min_words=1, max_words=len(KEYWORDS), style="Disney"):
    num_words = random.randint(min_words, min(max_words+1, len(KEYWORDS)))
    selected_keywords_lst = random.sample(KEYWORDS, num_words)
    style_keyword = random.sample(STYLE_TOKENS[style], 1)[0]
    indicator = random.randint(0, 4)
    if indicator == 2:
        style_keyword.replace(" ", "-")
    elif indicator == 3:
        style_keyword.replace(" ", "")

    selected_keywords_lst.append(style_keyword)
    random.shuffle(selected_keywords_lst)

    # Concatenate the keywords with a space in between
    selected_keywords = ", ".join(selected_keywords_lst)

    inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return f"{caption}; {selected_keywords}"



def generate_metadata(directory, multi_folder=False):
    output_file = os.path.join(directory, "metadata.jsonl")
    if not multi_folder:
        style = extract_style_name(directory)
    with open(output_file, 'w') as f:
        for filename in os.listdir(directory):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue
            if multi_folder:
                style = extract_style_name(filename)
            image_path = os.path.join(directory, filename)
            try:
                image = Image.open(image_path)
                caption = generate_caption(image, min_words=2, max_words=4, style=style)
                json_line = json.dumps({
                    "file_name": filename,
                    "text": caption
                })
                f.write(json_line + '\n')
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

    print(f"Metadata has been written to {output_file}")


def extract_style_name(folder):
    if folder.startswith("Disney"):
        return "Disney"
    elif folder.startswith("Pixar"):
        return "Pixar"
    elif folder.startswith("Japan"):
        return "Anime"
    return "Animation"