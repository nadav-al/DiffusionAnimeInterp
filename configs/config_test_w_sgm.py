# prefix = 'D:/University/ForthYear/GuidedProject/atd_12k/'
# prefix = 'Datasets_root/'
prefix = ''
testset_root = prefix + 'Datasets_root/test_2k_540p'
test_flow_root = prefix + 'Datasets_root/test_2k_pre_calc_sgm_flows'
test_annotation_root = prefix + 'Datasets_root/test_2k_annotations/'

# prefix = 'D:/University/ForthYear/GuidedProject/other_animations'
# testset_root = 'other_animations/full_shots/'
# test_flow_root = 'outputs/AoT_sgm_flows/'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std = [1, 1, 1]

inter_frames = 3

model = 'AnimeInterpNoCupy'
pwc_path = None

checkpoint = 'checkpoints/anime_interp_full.ckpt'

# store_path = 'outputs/avi_full_results'
store_path = 'outputs/test_multi_frames'



