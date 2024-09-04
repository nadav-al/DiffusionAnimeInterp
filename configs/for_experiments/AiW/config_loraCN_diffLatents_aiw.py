prefix = 'Datasets_root/'
testset_root = prefix + 'multiscenes/AliceInWonderland1/frames'
test_flow_root = prefix + 'datasets/test_2k_pre_calc_sgm_flows'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std = [1, 1, 1]

seed = 0

inter_frames = 1

model = 'LoraCNInterp'
# lora_weights_path = "checkpoints/outputs/LoRAs/07-24/test1/test1_07-22.json"
lora_weights_path = None
do_training = True
multiscene = True
pwc_path = None

checkpoint = 'checkpoints/anime_interp_full.ckpt'
diff_path = 'checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0'
controlnet_id = 'checkpoints/diffusers/diffusers/controlnet-canny-sdxl-1.0'


diff_objective = 'latents'
# diff_objective = None

ort = 'experiments/outputs/AIW/'
from os.path import join
store_path = join(ort, model, diff_objective)



