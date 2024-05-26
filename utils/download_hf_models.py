from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

ckpt_path = 'checkpoints/diffusers/'
adapters_path = ckpt_path + 'adapters/'
models_id = ['runwayml/stable-diffusion-v1-5',
             'stabilityai/stable-diffusion-xl-base-1.0']
adapters_id = ['h94/IP-Adapter']

for model_id in models_id:
    model = AutoPipelineForText2Image.from_pretrained(model_id, use_safetensors=True, variant="fp16")
    model.save_pretrained(ckpt_path + model_id, from_pt=True)