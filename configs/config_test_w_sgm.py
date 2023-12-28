prefix = 'D:/University/ForthYear/GuidedProject/atd_12k/'
# testset_root = prefix + 'datasets/test_2k_540p'
test_flow_root = prefix + 'datasets/test_2k_pre_calc_sgm_flows'
test_annotation_root = prefix + 'datasets/test_2k_annotations/'

prefix = 'D:/University/ForthYear/GuidedProject/other_animations'
testset_root = 'D:\University\ForthYear\GuidedProject\other_animations\datasets\Attack_On_Titan\AoT_s4e4_001_v1'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std  = [1, 1, 1]

inter_frames = 1

model = 'AnimeInterpNoCupy'
pwc_path = None

checkpoint = 'checkpoints/anime_interp_full.ckpt'

store_path = 'outputs/avi_full_results'



