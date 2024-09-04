prefix = 'Datasets_root/'
testset_root = prefix + 'multiscenes/AttackOnTitan/frames'
test_flow_root = prefix + 'multiscenes/AttackOnTitan/flows'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std = [1, 1, 1]

seed = 0

inter_frames = 1

model = 'LoraInterp'
# lora_weights_path = "checkpoints/outputs/LoRAs/07-24/test1/test1_07-22.json"
lora_weights_path = None
do_training = True
multiscene = True
pwc_path = None

checkpoint = 'checkpoints/anime_interp_full.ckpt'
diff_path = 'checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0'
# diff_path = 'checkpoints/diffusers/runwayml/stable-diffusion-v1-5'
# diff_path = 'CompVis/stable-diffusion-v1-4'



diff_objective = 'result'
# diff_objective = None

out_root = 'experiments/outputs/AoT/Lora/'
out_base = 'Results'

ort = 'experiments/outputs/AoT/'
from os.path import join
store_path = join(ort, model, diff_objective)



