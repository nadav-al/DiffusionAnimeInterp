import os.path

from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, ControlNetModel
import torch


ckpt_path = 'checkpoints/diffusers/'
adapters_path = ckpt_path + 'adapters/'
# models_id = ['runwayml/stable-diffusion-v1-5',
#              'stabilityai/stable-diffusion-xl-base-1.0',
#              'xinsir/controlnet-canny-sdxl-1.0'
#              ]
# models_id = ['diffusers/controlnet-canny-sdxl-1.0']
models_id = ['cagliostrolab/animagine-xl-3.1']
adapters_id = ['h94/IP-Adapter']

for id in models_id:
    if os.path.exists(ckpt_path + id):
        continue
    model = AutoPipelineForImage2Image.from_pretrained(id, use_safetensors=True)
    model.save_pretrained(ckpt_path + id, from_pt=True)
    # controlnet = ControlNetModel.from_pretrained(
    #     id, torch_dtype=torch.float16,
    #     variant="fp16", use_safetensors=True,
    # )
    # controlnet.save_pretrained(ckpt_path + id, from_pt=True)

# for id in adapters_id:
#     if os.path.exists(adapters_path + id):
#         continue
#     model = AutoPipelineForText2Image.from_pretrained(id, use_safetensors=True, variant="fp16")
#     model.save_pretrained(ckpt_path + id, from_pt=True)