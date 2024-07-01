import os
import random
from datetime import datetime
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

def generate_folder(folder_name, root_path="checkpoints/outputs/LoRAs", unique_folder=False):
    # Get current date in mm-dd format
    current_date = datetime.now().strftime("%m-%d")
    # Get the next test number
    test_num = get_next_test_number(root_path, current_date)
    if not unique_folder and test_num > 1:
        test_num -= 1

    # Create the full path
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

def random_size_crop(img, target_size, min_crop_ratio, max_crop_ratio, num_crops):
    width, height = img.size
    crop_width, crop_height = target_size
    crops = []
    for _ in range(num_crops):
        crop_ratio = random.randint(int(min_crop_ratio*10), int(max_crop_ratio*10))
        aug_crop_width = int(crop_width * (crop_ratio/10))
        aug_crop_height = int(crop_height * (crop_ratio/10))
        if aug_crop_height <= height and aug_crop_width <= width:
            left = random.randint(0, width - aug_crop_width)
            top = random.randint(0, height - aug_crop_height)
            cropped = img.crop((left, top, left+aug_crop_width, top+aug_crop_height))
        else:
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            cropped = img.crop((left, top, left + crop_width, top + crop_height))
        resized = cropped.resize(target_size, Image.LANCZOS)
        crops.append(resized)
    return crops

def sliding_window(img, size, overlap=0.5, jitter_range=0.1):
    width, height = img.size
    crop_width, crop_height = size
    step_x = int(crop_width * (1 - overlap))
    step_y = int(crop_height * (1 - overlap))
    crops = []
    for y in range(0, height - crop_height + 1, step_y):
        for x in range(0, width - crop_width + 1, step_x):
            jitter_x = int(random.uniform(-jitter_range, jitter_range) * step_x)
            jitter_y = int(random.uniform(-jitter_range, jitter_range) * step_y)
            left = max(0, min(x + jitter_x, width - crop_width))
            top = max(0, min(y + jitter_y, height - crop_height))
            croped = img.crop((left, top, left + crop_width, top + crop_height))
            resized = croped.resize(size, Image.LANCZOS)
            crops.append(resized)
    return crops

def jitter_random_crop(img, size, min_crop_ratio=0.6, max_crop_ratio=1.3, overlap=0.36, jitter_range=0.25):
    width, height = img.size
    crop_width, crop_height = size
    step_x = int(crop_width * (1 - overlap))
    step_y = int(crop_height * (1 - overlap))
    crops = []
    margin = int(height * jitter_range / 3)
    for y in range(margin, height - (crop_height + margin - 1), step_y):
        print()
        for x in range(margin, width - (crop_width + margin - 1), step_x):
            # Apply random jitter within the step range
            jitter_x = int(random.uniform(-jitter_range, jitter_range) * step_x)
            jitter_y = int(random.uniform(-jitter_range, jitter_range) * step_y)

            # Adjust for potential boundary violations due to jitter
            left = max(0, min(x + jitter_x, width - crop_width))
            top = max(0, min(y + jitter_y, height - crop_height))

            # Randomly determine crop size within the specified ratio range
            crop_ratio = random.randint(int(min_crop_ratio*10), int(max_crop_ratio*10))
            aug_crop_width = int(crop_width * crop_ratio)
            aug_crop_height = int(crop_height * crop_ratio)

            aug_left = left - abs(aug_crop_width-crop_width)/2
            aug_top = top - abs(aug_crop_height-crop_height)/2

            # Ensure crop stays within image boundaries
            if aug_left >= 0 and (aug_left + aug_crop_width) <= width and \
                    aug_top >= 0 and (aug_top + aug_crop_height) <= height:
                left = aug_left
                top = aug_top
                crop_width = aug_crop_width
                crop_height = aug_crop_height

            # Extract the random crop with jitter
            crop = img.crop((left, top, left + crop_width, top + crop_height))
            print(f"left={left}, top={top}, right={left+crop_width}, bottom={top+crop_height}")

            # Resize the crop to the final desired size
            resized_crop = crop.resize(size, Image.LANCZOS)
            crops.append(resized_crop)
    print()
    print()
    return crops


def calculate_color_variance(image: Image.Image) -> float:
    # Convert PIL Image to numpy array
    np_image = np.array(image)
    return np.sum(np.var(np_image, axis=(0, 1)))


def filter_crops(crops: List[Tuple[Image.Image, Tuple[int, int]]], threshold: float) -> List[Tuple[Image.Image, Tuple[int, int]]]:
    return [crop for crop in crops if calculate_color_variance(crop[0]) > threshold]

def preprocess(images_folder, size=(512, 512), min_ratio=0.7, max_ratio=1.3, overlap=0.3, num_crops=4, methods=None, unique_folder=False):
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
            g_size = (int(img.height * 0.4), int(img.height * 0.4))
            threshold = calculate_color_variance(img)/10
            crops = []
            for method in methods:
                if method == Methods.RANDOM_CROP:
                    crops += random_size_crop(img, size, 1.0, 1.0, num_crops)
                elif method == Methods.RANDOM_SIZE:
                    crops += random_size_crop(img, size, min_ratio, max_ratio, num_crops)
                elif method == Methods.GRID:
                    crops += jitter_random_crop(img, g_size, 1.0, 1.0, jitter_range=0, overlap=overlap)
                elif method == Methods.JITTER:
                    crops += jitter_random_crop(img, g_size, 1.0, 1.0, overlap=overlap)
                elif method == Methods.GRID_RANDOM:
                    crops += jitter_random_crop(img, g_size, min_ratio, max_ratio, jitter_range=0, overlap=overlap)
                elif method == Methods.JITTER_RANDOM:
                    crops += jitter_random_crop(img, g_size, min_ratio, max_ratio, overlap=overlap)
                else:
                    raise ValueError(f"Unknown method: {method}")

            filtered_crops = filter_crops(crops, threshold)
            print((len(crops) - len(filtered_crops)), " crops where removed because of threshold")
            for i, crop in enumerate(filtered_crops):
                output_path = os.path.join(output_folder,
                                           f"{os.path.splitext(image_name)[0]}_crop{i + 1}.jpg")
                crop.save(output_path)

    return output_folder

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt")

    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption + " in the style of AniInt"



def generate_metadata(directory):
    output_file = os.path.join(directory, "metadata.jsonl")
    with open(output_file, 'w') as f:
        length = len(os.listdir(directory))
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(directory, filename)
                try:
                    # caption = generate_caption(image_path)
                    caption = "a photo in the style of AniInt"
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