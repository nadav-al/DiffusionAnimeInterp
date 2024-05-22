import argparse
import os

import models
import datas
import torch
import torchvision.transforms as TF
import torch.nn as nn
import sys
import cv2
from utils.vis_flow import flow_to_color
from utils.config import Config

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

def save_flow_to_img(flow, des):
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(des + '.jpg', cf)


def validate(config):
    # preparing datasets & normalization
    normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
    normalize2 = TF.Normalize([0, 0, 0], config.std)
    trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

    revmean = [-x for x in config.mean]
    revstd = [1.0 / x for x in config.std]
    revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
    revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
    revNormalize = TF.Compose([revnormalize1, revnormalize2])
    revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

#     refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )



    testset = datas.AniTripletWithSGMFlowTest(config.testset_root, config.test_flow_root, trans, config.test_size,
                                              config.test_crop_size, train=False)
    sampler = torch.utils.data.SequentialSampler(testset)
    validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)
    to_img = TF.ToPILImage()

    print(testset)
    sys.stdout.flush()

    # prepare model
    if torch.cuda.is_available():
        model = getattr(models, config.model)(config.pwc_path).cuda()
    else:
        model = getattr(models, config.model)(config.pwc_path, config=config)
    model = nn.DataParallel(model)
    retImg = []

    # load weights
    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'], strict=False)

    # prepare others
    store_path = config.store_path

    folders = []

    print('Everything prepared. Ready to test...')
    sys.stdout.flush()

    #  start testing...
    with torch.no_grad():
        model.eval()
        ii = 0
        for validationIndex, validationData in enumerate(validationloader, 0):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testset)))
            sys.stdout.flush()
            sample, flow, index, folder = validationData

            frame0 = None
            frame1 = sample[0]
            frame3 = None
            frame2 = sample[-1]

            folders.append(folder[0][0])

            # initial SGM flow
            F12i, F21i = flow

            F12i = F12i.float()
            F21i = F21i.float()

            ITs = [sample[tt] for tt in range(1, 2)]
            I1 = frame1
            I2 = frame2

            if not os.path.exists(config.store_path + '/' + folder[0][0]):
                os.mkdir(config.store_path + '/' + folder[0][0])

            revtrans(I1.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[0][0] + '.png')
            revtrans(I2.cpu()[0]).save(store_path + '/' + folder[-1][0] + '/' + index[-1][0] + '.png')
            for tt in range(config.inter_frames):
                x = config.inter_frames
                t = 1.0 / (x + 1) * (tt + 1)

                outputs = model(I1, I2, F12i, F21i, t)

                It_warp = outputs[0]

                warp_img = to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))
                # warp_img = refiner_pipe(image=warp_img).images[0]
                warp_img.save(store_path + '/' + folder[1][0] + '/' + index[1][0] + '.png')

                save_flow_to_img(outputs[1].cpu(), store_path + '/' + folder[1][0] + '/' + 'F12_' + index[1][0] )
                save_flow_to_img(outputs[2].cpu(), store_path + '/' + folder[1][0] + '/' + 'F21_' + index[1][0] )


if __name__ == "__main__":

    # loading configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)

    if not os.path.exists(config.store_path):
        os.mkdir(config.store_path)

    validate(config)