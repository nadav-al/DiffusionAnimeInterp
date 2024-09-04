prefix = 'Datasets_root/'
testset_root = prefix + 'multiscenes/AnimationTest/frames'
test_flow_root = prefix + 'multiscenes/AnimationTest/flows'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std = [1, 1, 1]

seed = 0

inter_frames = 1

model = 'AnimeInterp'
pwc_path = None

checkpoint = 'checkpoints/anime_interp_full.ckpt'

ort = 'experiments/outputs/SAT'
from os.path import join
store_path = join(ort, model)



