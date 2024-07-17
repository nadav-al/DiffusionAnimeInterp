from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import os
from PIL import Image
from utils.lora_utils import generate_caption
import torch

diff_path = "checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0"
# diff_path = "checkpoints/diffusers/cagliostrolab/animagine-xl-3.1"
# diff_path = "checkpoints/diffusers/runwayml/stable-diffusion-v1-5"
lora_path = "sd-model-fintuned-lora/07-16/test6/sdxl/full_shots"
animeinterp_path = "outputs/old_tests/test_aot_DiffimeInterp_latents"
weight_name = "pytorch_lora_weights.safetensors"

output_folder = "outputs/multifolder_lora/07-16/test6"

ns = 20  # Num inference steps


if __name__ == '__main__':
    pipeline = AutoPipelineForText2Image.from_pretrained(diff_path, torch_dtype=torch.float16).to("cuda")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ckpt_dirs = os.listdir(lora_path)


    for image_folder in os.listdir(animeinterp_path):
        image_path = os.path.join(animeinterp_path, image_folder, "frame2.png")
        image = Image.open(image_path)
        image = image.resize((1920, 1080))
        caption = generate_caption(image, max_words=3, style="Anime")
        output_path = os.path.join(output_folder, image_folder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for ckpt in ckpt_dirs:
            ckpt_path = os.path.join(lora_path, ckpt)
            if os.path.isfile(ckpt_path):
                pipeline.load_lora_weights(ckpt_path)
                output_filename = "full_shots"
            else:
                pipeline.load_lora_weights(os.path.join(ckpt_path, "pytorch_lora_weights.safetensors"))
                output_filename = ckpt

            output = pipeline(caption,
                              width=image.width, height=image.height,
                              image=image, num_inference_steps=ns,
                              negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details. hyperrealistic. 3D").images[0]

            output.save(os.path.join(output_path, f"{output_filename}.png"))
            pipeline.unload_lora_weights()









