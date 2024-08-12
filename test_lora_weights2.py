from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import os
from PIL import Image
from utils.captionning import generate_caption
import torch

diff_path = "checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0"
# diff_path = "checkpoints/diffusers/cagliostrolab/animagine-xl-3.1"
# diff_path = "checkpoints/diffusers/runwayml/stable-diffusion-v1-5"
lora_path = "checkpoints/outputs/LoRAs/08-10/test2/rank8"
weight_name = "pytorch_lora_weights.safetensors"

output_path = "outputs/weights_tests/07-17/sdxl/test2"

ns = 25  # Num inference steps


def get_image_path(scene):
    return f"outputs/old_tests/test_aot_DiffimeInterp_latents/{scene}/frame2.png"


if __name__ == '__main__':
    pipeline = AutoPipelineForText2Image.from_pretrained(diff_path, torch_dtype=torch.float16).to("cuda")
    # pipeline = AutoPipelineForText2Image.from_pretrained(diff_path).to("cuda")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # root_name = "Disney_v4_21_044474_s1"
    # root_path = os.path.join(lora_path, root_name)
    # image = Image.open(animeinterp_output_path)
    # image = Image.open("temp_img.png")
    for root_name in os.listdir(lora_path):
    # for i in range(0, 1):
        print(f"Processing {root_name}")
        root_path = os.path.join(lora_path, root_name)
        image = Image.open(get_image_path(root_name))
        image = image.resize((1920, 1080))
        folder_list = os.listdir(root_path)
        folder_list.sort()
        output_root_folder = os.path.join(output_path, root_name)
        if not os.path.exists(output_root_folder):
            os.makedirs(output_root_folder)
        caption = generate_caption(image, max_words=3, style="Anime")
        print(folder_list)
        for idx, folder_name in enumerate(folder_list):
            if idx > 86:
                break
            folder_path = os.path.join(root_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Check if it's a directory
            if idx%10 == 0:
                print(f"Processing {folder_name}")
            # Construct the path to the LoRA weights file
            lora_weights_path = folder_path
            # lora_weights_path = os.path.join(folder_path, "pytorch_lora_weights.safetensors")
            # Check if the LoRA weights file exists
            if os.path.exists(lora_weights_path):
            # if False:
                try:
                    # Load the LoRA weights
                    pipeline.load_lora_weights(lora_weights_path)
                    # print(f"  - Loaded LoRA weights from {lora_weights_path}")
                except Exception as e:
                    print(f"Error processing {folder_name}: {str(e)}")
            else:
                print(f"No LoRA weights found in {folder_name}")
            for step in range(5, ns, 5):
                output = pipeline(caption,
                                  width=image.width, height=image.height,
                                  image=image, num_inference_steps=step, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details. hyperrealistic. 3D").images[0]
                if not os.path.exists(os.path.join(output_root_folder, folder_name)):
                    os.makedirs(os.path.join(output_root_folder, folder_name))
                output.save(os.path.join(output_root_folder, folder_name, f"{step}_steps.png"))
