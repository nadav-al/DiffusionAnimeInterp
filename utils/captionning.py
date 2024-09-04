import os
import random
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import json
from .files_and_folders import extract_style_name

custom_cache_dir = "/cs/labs/danix/nadav_al/AnimeInterp/checkpoints/Blip"
os.environ['HF_HOME'] = custom_cache_dir
#
# Set up the BLIP image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=custom_cache_dir)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=custom_cache_dir, torch_dtype=torch.float16).to("cuda")

UNIQUE_TOKEN = 'dinep'
KEYWORDS = {
    'best quality',
    'high quality',
    'cel shading',
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
    # '2D',
}


STYLE_TOKENS = {
    'Disney': ['Disney-Renaissance', f'{UNIQUE_TOKEN} Disney', 'Disney', '2D', f'2D {UNIQUE_TOKEN}'],
    'Anime': ['Anime', 'StudioGhibli', f'{UNIQUE_TOKEN} Anime', f'Ghibli {UNIQUE_TOKEN}', '2D', f'2D {UNIQUE_TOKEN}'],
    'Pixar': ['Pixar', f'{UNIQUE_TOKEN} Pixar', '3D', f'3D {UNIQUE_TOKEN}'],
    'Animation': ['Animation', f'{UNIQUE_TOKEN} Animation', f'{UNIQUE_TOKEN} Drawing']
}


def generate_keywords(style="Animation", min_words=1, max_words=len(KEYWORDS)):
    num_words = random.randint(min_words, min(max_words + 1, len(KEYWORDS)))
    selected_keywords_lst = random.sample(KEYWORDS, num_words)
    style_keyword = random.choice(STYLE_TOKENS[style])
    selected_keywords_lst.append(style_keyword)
    random.shuffle(selected_keywords_lst)

    seperator = "; "
    idx = random.randint(0, 5)
    if idx == 3:
        seperator = ". "
    if idx == 4:
        seperator = ", "

    return seperator.join(selected_keywords_lst)

def generate_caption(image, style="Animation", min_words=1, max_words=len(KEYWORDS)):
    selected_keywords = generate_keywords(style, min_words, max_words)

    inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return f"{caption}; {selected_keywords}"


def generate_metadata(directory, with_caption=True):
    output_file = os.path.join(directory, "metadata.jsonl")
    style = extract_style_name(directory)
    print(directory, style)
    with open(output_file, 'w') as f:
        for filename in os.listdir(directory):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue
            image_path = os.path.join(directory, filename)
            try:
                image = Image.open(image_path)
                if with_caption:
                    caption = generate_caption(image, style, min_words=2, max_words=4)
                else:
                    caption = generate_keywords(style, min_words=2, max_words=4)

                json_line = json.dumps({
                    "file_name": filename,
                    "text": caption
                })
                f.write(json_line + '\n')
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

    print(f"Metadata has been written to {output_file}")