import argparse
import os

import models
import datas
import torch
import torchvision.transforms as TF
import torch.nn as nn
import sys
import cv2

from models import DiffimeInterp
from utils.vis_flow import flow_to_color
from utils.config import Config
from utils.lora_utils import generate_folder

import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

def save_flow_to_img(flow, des):
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(des + '.jpg', cf)


def validate(config, args):
    # preparing datasets & normalization
    normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
    normalize2 = TF.Normalize([0, 0, 0], config.std)
    trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

    revmean = [-x for x in config.mean]
    revstd = [1.0 / x for x in config.std]
    revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
    revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
    revNormalize = TF.Compose([revnormalize1, revnormalize2])


    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    testset = datas.AniTripletWithSGMFlowTest(config.testset_root, config.test_flow_root, trans, config.test_size,
                                              config.test_crop_size, train=False)
    sampler = torch.utils.data.SequentialSampler(testset)

    validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)
    to_img = TF.ToPILImage()

    sys.stdout.flush()

    # prepare model
    if config.model in [ 'AnimeInterp', 'AnimeInterpNoCupy' ]:
        model = getattr(models, config.model)(config.pwc_path).to(device)
    elif config.model == "LoraInterp":
        base_model = DiffimeInterp(config.pwc_path, config=config).to(device)
        model = getattr(models, config.model)(base_model).to(device)
    else:
        model = getattr(models, config.model)(config.pwc_path, config=config).to(device)
    model = nn.DataParallel(model)

    lora_train_flag = False
    if config.model == "LoraInterp":
        if config.lora_weights is not None:
            lora_train_flag = True
            weights_path = config.lora_weights
        elif config.multifolder:
            lora_train_flag = True
            weights_path = model.module.train_lora_from_multi_folder(config.testset_root)

    # load weights
    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'], strict=False)

    # prepare others
    folders = []

    print('Everything prepared. Ready to test...')
    sys.stdout.flush()

    #  start testing...
    with torch.no_grad():
        model.eval()
        ii = 0
        for validationIndex, validationData in enumerate(validationloader, 0):
            print('Testing {}/{}-th group...'.format(validationIndex+1, len(testset)))
            sys.stdout.flush()
            unique_folder = validationIndex == 0
            sample, flow, index, folder = validationData
            first_frame = sample[0]
            last_frame = sample[-1]

            folders.append(folder[0][0])

            # initial SGM flow
            F12i, F21i = flow
            F12i = F12i.float().to(device)
            F21i = F21i.float().to(device)
            I1 = first_frame.to(device)
            I2 = last_frame.to(device)

            num_of_frames = config.inter_frames

            store_path = generate_folder(folder[0][0], config.store_path, unique_folder=unique_folder)

            # save the first and last frame
            print(store_path, os.path.exists(store_path))

            if config.model == 'LoraInterp':
                if not lora_train_flag:
                    weights_path = model.module.train_lora_from_single_folder(folder[0][0], unique_folder=unique_folder)

            x = num_of_frames
            for tt in range(num_of_frames):
                t = 1.0 / (x + 1) * (tt + 1)
                if config.model in [ 'AnimeInterp', 'AnimeInterpNoCupy' ]:
                    outputs = model(I1, I2, F12i, F21i, t)
                elif config.model == 'LoraInterp':
                    outputs = model(I1, I2, F12i, F21i, t, folder=folder, weights_path=weights_path)
                else:
                    outputs = model(I1, I2, F12i, F21i, t, folder=folder)

                if outputs is None:
                    continue

                It_warp = outputs[0]

                warp_img = to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))
                warp_img.save(os.path.join(store_path, folder[1][0], f'frame{tt+2}.png'))

                if tt == 0:
                    save_flow_to_img(outputs[1].cpu(), os.path.join(store_path, folder[1][0], 'flows/F12'))
                    save_flow_to_img(outputs[2].cpu(), os.path.join(store_path, folder[1][0], 'flows/F21'))


if __name__ == "__main__":

    # loading configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    # parser.add_argument('lora_config', default='configs/LoRA/lora_config_base.py')
    args, _ = parser.parse_known_args()
    config = Config.from_file(args.config)
    # lora_config = Config.from_file(args.lora_config)
    lora_config=None

    if not os.path.exists(config.store_path):
        os.makedirs(config.store_path)

    validate(config, args)