import os
import random
from datetime import datetime

import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, PaliGemmaForConditionalGeneration
import numpy as np
import json


# custom_cache_dir = "/cs/labs/danix/nadav_al/AnimeInterp/checkpoints/Blip"
# os.environ['HF_HOME'] = custom_cache_dir
#
# # Set up the BLIP image captioning model
# processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", cache_dir=custom_cache_dir, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", cache_dir=custom_cache_dir, trust_remote_code=True)
# model = PaliGemmaForConditionalGeneration.from_pretrained("microsoft/Florence-2-large", cache_dir=custom_cache_dir, trust_remote_code=True)

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

def generate_folder(folder_name, root_path="checkpoints/outputs/LoRAs", extension=None, unique_folder=False):
    # Get current date in mm-dd format
    current_date = datetime.now().strftime("%m-%d")
    # Get the next test number
    test_num = get_next_test_number(root_path, current_date)
    if not unique_folder and test_num > 1:
        test_num -= 1

    # Create the full path
    if extension:
        full_path = os.path.join(root_path, current_date, f"test{test_num}", extension, folder_name)
    else:
        full_path = os.path.join(root_path, current_date, f"test{test_num}", folder_name)
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

def choose_to_use_scaled():
    return bool(random.getrandbits(1)) and bool(random.getrandbits(1))

def random_size_crop(img, size, target_size, min_crop_ratio, max_crop_ratio, num_crops):
    width, height = img.size
    crop_width, crop_height = size
    crops = []
    for _ in range(num_crops):
        crop_ratio = random.randint(int(min_crop_ratio*10), int(max_crop_ratio*10))
        aug_crop_width = int(crop_width * (crop_ratio/10))
        aug_crop_height = int(crop_height * (crop_ratio/10))
        if aug_crop_height <= height and aug_crop_width <= width:
            left = random.randint(0, width - aug_crop_width)
            top = random.randint(0, height - aug_crop_height)
            croped = img.crop((left, top, left+aug_crop_width, top+aug_crop_height))
        else:
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            croped = img.crop((left, top, left + crop_width, top + crop_height))
        resized = croped.resize((target_size, target_size), Image.LANCZOS)
        crops.append(resized)
    return crops


def jitter_random_crop(img, size, target_size, min_crop_ratio=0.6, max_crop_ratio=1.3, overlap=0.36, jitter_range=0.15):
    width, height = img.size
    scimg = img.resize((target_size, target_size))
    crop_width, crop_height = size
    step_x = int(crop_width * (1 - overlap))
    step_y = int(crop_height * (1 - overlap))
    crops = []
    margin = int(height * jitter_range / 7)
    for y in range(margin, height - (crop_height + margin - 1), step_y):
        for x in range(margin, width - (crop_width + margin - 1), step_x):
            # Apply random jitter within the step range
            jitter_x = int(random.uniform(-jitter_range, jitter_range) * step_x)
            jitter_y = int(random.uniform(-jitter_range, jitter_range) * step_y)

            # Ensure crop stays within image boundaries
            if choose_to_use_scaled():
                # Adjust for potential boundary violations due to jitter
                left = max(0, min(x + jitter_x, target_size - crop_width))
                top = max(0, min(y + jitter_y, target_size - crop_height))

                # Randomly determine crop size within the specified ratio range
                crop_ratio = random.randint(int(min_crop_ratio * 10), int(max_crop_ratio * 10))
                aug_crop_width = int(crop_width * crop_ratio/15)
                aug_crop_height = int(crop_height * crop_ratio/15)

                aug_left = max(left - abs(aug_crop_width - crop_width) / 2, 0)
                aug_top = max(top - abs(aug_crop_height - crop_height) / 2, 0)
                if (aug_left + aug_crop_width) > target_size:
                    aug_crop_width = crop_width
                    if left + crop_width > target_size:
                        aug_crop_width /= (4/3)
                if (aug_top + aug_crop_height) > target_size:
                    aug_crop_height = crop_height
                    if top + crop_height > target_size:
                        aug_crop_height /= 2

                # Extract the random crop with jitter
                crop = scimg.crop((aug_left, aug_top, aug_left + aug_crop_width, aug_top + aug_crop_height))
            else:
                # Adjust for potential boundary violations due to jitter
                left = max(0, min(x + jitter_x, width - crop_width))
                top = max(0, min(y + jitter_y, height - crop_height))

                # Randomly determine crop size within the specified ratio range
                crop_ratio = random.randint(int(min_crop_ratio * 10), int(max_crop_ratio * 10))
                aug_crop_width = int(crop_width * crop_ratio/10)
                aug_crop_height = int(crop_height * crop_ratio/10)

                aug_left = max(left - abs(aug_crop_width - crop_width) / 2, 0)
                aug_top = max(top - abs(aug_crop_height - crop_height) / 2, 0)

                if (aug_left + aug_crop_width) > width:
                    aug_crop_width = crop_width
                if (aug_top + aug_crop_height) > height:
                    aug_crop_height = crop_height

                # Extract the random crop with jitter
                crop = img.crop((aug_left, aug_top, aug_left + aug_crop_width, aug_top + aug_crop_height))

            # Resize the crop to the final desired size
            resized_crop = crop.resize((target_size, target_size), Image.LANCZOS)
            crops.append(resized_crop)
    return crops


def calculate_color_variance(image):
    # Convert PIL Image to numpy array
    np_image = np.array(image)
    max_variance = (255.0**2)/10
    return np.median(np.var(np_image, axis=(0, 1), ddof=1)/max_variance)


def filter_crops(crops, threshold):
    f_crops = []
    acc = []
    dec = []
    for crop in crops:
        var = calculate_color_variance(crop)
        if var > threshold:
            f_crops.append(crop)
            acc.append(var)
        else:
            dec.append(var)
    # print(f"accepted crops' median var = {np.median(acc)}")
    # print(f"declined crops' median var = {np.median(dec)}")
    return f_crops

def preprocess(images_folder, size=(512, 512), target_size=512, min_ratio=0.6, max_ratio=1.3, overlap=0.5, num_crops=15, methods=None, unique_folder=False):
    output_folder = generate_folder(os.path.split(images_folder)[1], "TempDatasets", unique_folder=unique_folder)
    if methods is None:
        methods = [Methods.JITTER_RANDOM]
    image_files = os.listdir(images_folder)
    image_files.sort()
    for idx, image_name in enumerate(image_files):
        if idx >= 3 or idx == 1 or not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        image_path = os.path.join(images_folder, image_name)
        with Image.open(image_path) as img:
            s = int((3*img.height + 4*size[0])/7)
            g_size = (s, s)

            threshold = calculate_color_variance(img)/5
            crops = []
            for method in methods:
                if method == Methods.RANDOM_CROP:
                    crops += random_size_crop(img, g_size, target_size, 1.0, 1.0, int(2*num_crops/3))
                    crops += random_size_crop(img, size, target_size, 0.8, 0.9, int(num_crops/3))
                elif method == Methods.RANDOM_SIZE:
                    crops += random_size_crop(img, g_size, target_size, min_ratio, max_ratio, int(2*num_crops/3))
                    crops += random_size_crop(img, size, target_size, 0.8, 0.9, int(num_crops / 3))
                elif method == Methods.GRID:
                    crops += jitter_random_crop(img, g_size, target_size, 1.0, 1.0, jitter_range=0, overlap=overlap)
                elif method == Methods.JITTER:
                    crops += jitter_random_crop(img, g_size, target_size, 1.0, 1.0, overlap=overlap)
                elif method == Methods.GRID_RANDOM:
                    crops += jitter_random_crop(img, g_size, target_size, min_ratio, max_ratio, jitter_range=0, overlap=overlap)
                elif method == Methods.JITTER_RANDOM:
                    crops += jitter_random_crop(img, g_size, target_size, min_ratio, max_ratio, overlap=overlap)
                else:
                    raise ValueError(f"Unknown method: {method}")

            f_crops = filter_crops(crops, threshold)
            print((len(crops) - len(f_crops)), " crops out of ", len(crops), " where removed because of threshold")
            for i, crop in enumerate(f_crops):
                output_path = os.path.join(output_folder,
                                           f"{os.path.splitext(image_name)[0]}_crop{i + 1}.jpg")
                crop.save(output_path)

    return output_folder



# def generate_caption(image_path):
#     image = Image.open(image_path).convert('RGB')
#     inputs = processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt")
#
#     outputs = model.generate(**inputs)
#     caption = processor.decode(outputs[0], skip_special_tokens=True)
#
#     return caption + " in the style of AniInt"

UNIQUE_TOKEN = 'dinep'

KEYWORDS = {
    'best quality',
    'cel shading',
    f'{UNIQUE_TOKEN} Anime',
    # 'Anime',
    # 'Disney-Renaissance',
    f'{UNIQUE_TOKEN} Disney',
    'animation',
    'drawing',
    'Cartoon',
    'Classic Cartoon',
    'Hard Shadows',
    'Vibrant Colors',
    'Flat Color',
    'Line Art',
    'Frame-by-Frame',
    'clear lines',
    'Extreme Poses',
    'Hand-Drawn',
    '2D',
}

def generate_caption(min_words=1, max_words=len(KEYWORDS)):
    num_words = random.randint(min_words, min(max_words, len(KEYWORDS)))
    selected_keywords = random.sample(KEYWORDS, num_words)

    # Concatenate the keywords with a space in between
    caption = ", ".join(selected_keywords)
    return f"high quality, {UNIQUE_TOKEN} style, " + caption



def generate_metadata(directory):
    output_file = os.path.join(directory, "metadata.jsonl")
    with open(output_file, 'w') as f:
        length = len(os.listdir(directory))
        for filename in os.listdir(directory):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue
            image_path = os.path.join(directory, filename)
            try:
                caption = generate_caption(min_words=2, max_words=4)
                # caption = "high quality, best quality, anime, cartoon, hand-drawn, animation, Disney"
                json_line = json.dumps({
                    "file_name": filename,
                    "text": caption
                })
                f.write(json_line + '\n')
                if length < 6:
                    f.write(json_line + '\n')
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

    print(f"Metadata has been written to {output_file}")

