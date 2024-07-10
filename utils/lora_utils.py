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

def generate_folder(folder_name, root_path="checkpoints/outputs/LoRAs", extension=None, unique_folder=False):
    # Get current date in mm-dd format
    current_date = datetime.now().strftime("%m-%d")
    # Get the next test number
    test_num = get_next_test_number(root_path, current_date)
    if not unique_folder and test_num > 1:
        test_num -= 1

    # Create the full path
    # full_path = os.path.join(root_path, current_date, f"test{test_num}", folder_name)
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

def random_size_crop(img, size, target_size, min_crop_ratio, max_crop_ratio, num_crops):
    width, height = img.size
    crop_width, crop_height = size
    crops = []
    for _ in range(num_crops):
        crop_ratio = random.randint(int(min_crop_ratio*10), int(max_crop_ratio*10))
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
    print()
    return crops


# def calculate_color_std(image):
#     # Convert PIL Image to numpy array
#     np_image = np.array(image)
#     mean = np.mean(np_image)
#     print(mean.shape)
#     # print(f"max color value = ", np.max(mean, axis=(0,1)), " ; min color value = ", np.min(mean, axis=(0,1)))
#     return np.sum(np.var(np_image, axis=(0, 1), ddof=1))/3

def calculate_color_std(image):
    img_array = np.array(image)

    # Reshape the image to a 2D array of pixels
    # pixels = img_array.reshape(-1, 3)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = img_array.astype(np.float32) / 255.0
    return np.mean(img_array)
    # Calculate the variance for each channel
    # std_r = np.std(img_array[:, 0])
    # std_g = np.std(img_array[:, 1])
    # std_b = np.std(img_array[:, 2])
    # # std_r = np.mean(pixels[:, 0])
    # # std_g = np.mean(pixels[:, 1])
    # # std_b = np.mean(pixels[:, 2])
    #
    # # Calculate the overall color variance (you can use different methods here)
    # # overall_std = np.sqrt((std_r**2 + std_g**2 + std_b**2))/3
    # overall_std = np.mean([std_r, std_g, std_b])
    # # print({
    # #     "red": std_r,
    # #     "green": std_g,
    # #     "blue": std_b,
    # #     "overall": overall_std
    # # })
    # return overall_std




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
    # print(f"image mean value = {img_mean}, with threshold = 1/{img_mean/threshold}")
    # print(f"accepted crops' mean value = {np.median(acc)}")
    # print(f"declined crops' mean value = {np.median(dec)}")
    return f_crops, d_crops

def preprocess(images_folder, size=(512, 512), target_size=512, min_ratio=0.7, max_ratio=1.3, overlap=0.3, num_crops=10, methods=None, unique_folder=False, save_folder=True):
    output_folder = ""
    if save_folder:
        output_folder = generate_folder(os.path.split(images_folder)[1], "TempDatasets", unique_folder=unique_folder)
    if methods is None:
        methods = [Methods.JITTER_RANDOM]
    image_files = os.listdir(images_folder)
    image_files.sort()
    sizes = [0.4, 0.55, 0.6, 0.65, 0.72, 0.78, 0.85, 0.9, 0.95, 1]
    repeats = [2, 2, 2, 3, 2, 3, 2, 2, 1, 1]
    for idx, image_name in enumerate(image_files):
        if idx >= 3 or idx == 1 or not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        image_path = os.path.join(images_folder, image_name)
        with Image.open(image_path) as img:
            # g_size = (int(img.height * 0.8), int(img.height * 0.7))
            img_mean = calculate_color_std(img)
            total_crops = []
            for threshold in range(7, 4, -1):
                crops = []
                for j in range(len(sizes)):
                    g_size = int(img.height*sizes[j])
                    for i in range(repeats[j]):
                        l1 = random.randint(0, img.width - g_size)
                        l2 = random.randint(0, img.width - g_size)
                        left = (0 + l1 + l2) // 3
                        t1 = random.randint(0, img.height - g_size)
                        t2 = random.randint(0, img.height - g_size)
                        t3 = random.randint(0, img.height - g_size)
                        top = (t1 + t2 + t3) // 3
                        cropped = img.crop((left, top, left+g_size, top+g_size))
                        # resized = cropped.resize((target_size, target_size))
                        # crops.append(resized)
                        crops.append(cropped)
                # for method in methods:
                #     if method == Methods.RANDOM_CROP:
                #         crops += random_size_crop(img, g_size, target_size, 1.0, 1.0, int(2 * num_crops / 3))
                #         crops += random_size_crop(img, size, target_size, 0.8, 0.9, int(num_crops / 3))
                #     elif method == Methods.RANDOM_SIZE:
                #         crops += random_size_crop(img, g_size, target_size, min_ratio, max_ratio, int(2 * num_crops / 3))
                #         crops += random_size_crop(img, size, target_size, 0.8, 0.9, int(num_crops / 3))
                #     elif method == Methods.GRID:
                #         crops += jitter_random_crop(img, size, target_size, 1.0, 1.0, jitter_range=0, overlap=overlap)
                #     elif method == Methods.JITTER:
                #         crops += jitter_random_crop(img, size, target_size, 1.0, 1.0, overlap=overlap)
                #     elif method == Methods.GRID_RANDOM:
                #         crops += jitter_random_crop(img, size, target_size, min_ratio, max_ratio, jitter_range=0, overlap=overlap)
                #     elif method == Methods.JITTER_RANDOM:
                #         crops += jitter_random_crop(img, size, target_size, min_ratio, max_ratio, overlap=overlap)
                #     else:
                #         raise ValueError(f"Unknown method: {method}")

                # f_crops, d_crops = filter_crops(crops, img_mean, img_mean/threshold)
                # print(len(f_crops), " crops remains out of ", len(crops))
                # total_crops += random.sample(crops, min(len(crops), num_crops))
                total_crops += crops
                # print("total crops for this frame: ", len(total_crops))
            # final_mean = np.mean([calculate_color_std(c) for c in total_crops])
            # final_std = np.std([calculate_color_std(c) for c in total_crops])
            # print("final mean = ", final_mean)
            # print("final std = ", final_std)
            # final_crops, _ = filter_crops(total_crops, final_mean, final_mean/10)
            # print(len(final_crops), " final crops remains out of ", len(total_crops), " total crops")
                # final_crops = random.sample(final_crops, min(len(final_crops), num_crops))
            for i, crop in enumerate(total_crops):
                output_path = os.path.join(output_folder,
                                           f"{os.path.splitext(image_name)[0]}_crop{i + 1}.png")

                crop.save(output_path)
            img.save(os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_1.png"))
            img.save(os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_2.png"))
    return output_folder


UNIQUE_TOKEN = 'dinep'
KEYWORDS = {
    'best quality',
    'cel shading',
    # f'{UNIQUE_TOKEN} Anime',
    # 'Anime',
    # 'Disney-Renaissance',
    # f'{UNIQUE_TOKEN} Disney',
    'colorfull',
    'animation',
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

def generate_caption(image, min_words=1, max_words=len(KEYWORDS), style="Disney"):
    num_words = random.randint(min_words, min(max_words+1, len(KEYWORDS)))
    selected_keywords_lst = random.sample(KEYWORDS, num_words)
    random.shuffle(selected_keywords_lst)

    # Concatenate the keywords with a space in between
    selected_keywords = ", ".join(selected_keywords_lst)

    inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return f"{caption}. high quality. {UNIQUE_TOKEN}-{style}-style, {selected_keywords}"



def generate_metadata(directory):
    output_file = os.path.join(directory, "metadata.jsonl")
    if directory.startswith("Disney"):
        style = "Disney"
    else:
        style = "Anime"
    with open(output_file, 'w') as f:
        length = len(os.listdir(directory))
        for filename in os.listdir(directory):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue
            image_path = os.path.join(directory, filename)
            try:
                image = Image.open(image_path)
                caption = generate_caption(image, min_words=2, max_words=4, style=style)
                print(filename, ": ", caption)
                json_line = json.dumps({
                    "file_name": filename,
                    "text": caption
                })
                f.write(json_line + '\n')
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

    print(f"Metadata has been written to {output_file}")