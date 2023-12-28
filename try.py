import models
import datas
import argparse
import torch
import torchvision.transforms as TF
import torch.nn as nn
import os
from utils.config import Config
import sys
import json
from test_anime_sequence_one_by_one import save_flow_to_img


def inference(config):
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

    testset = datas.AniTripletWithSGMFlowTest(config.testset_root, config.test_flow_root, trans, config.test_size,
                                              config.test_crop_size, train=False)
    sampler = torch.utils.data.SequentialSampler(testset)
    validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)
    to_img = TF.ToPILImage()
    model = getattr(models, config.model)(config.pwc_path)
    model = nn.DataParallel(model)

    # load weights
    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'], strict=False)

    # prepare others
    store_path = config.store_path

    folders = []

    with torch.no_grad():
        model.eval()
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

            revtrans(I1.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[0][0] + '.jpg')
            revtrans(I2.cpu()[0]).save(store_path + '/' + folder[-1][0] + '/' + index[-1][0] + '.jpg')
            for tt in range(config.inter_frames):
                x = config.inter_frames
                t = 1.0 / (x + 1) * (tt + 1)

                outputs = model(I1, I2, F12i, F21i, t)

                It_warp = outputs[0]

                to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)).save(
                    store_path + '/' + folder[1][0] + '/' + index[1][0] + '.png')

                save_flow_to_img(outputs[1].cpu(), store_path + '/' + folder[1][0] + '/' + index[1][0] + '_F12')
                save_flow_to_img(outputs[2].cpu(), store_path + '/' + folder[1][0] + '/' + index[1][0] + '_F21')

                # estimated = revNormalize(It_warp[0].cpu()).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
                # gt = revNormalize(ITs[tt][0]).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
                #
                # labelFilePath = os.path.join(config.test_annotation_root,
                #                              folder[1][0], '%s.json' % folder[1][0])
                #
                # # crop region of interest
                # with open(labelFilePath, 'r') as f:
                #     jsonObj = json.load(f)
                #     motion_RoI = jsonObj["motion_RoI"]
                #     level = jsonObj["level"]
                #
                # tempSize = jsonObj["image_size"]
                # scaleH = float(tempSize[1]) / config.test_size[1]
                # scaleW = float(tempSize[0]) / config.test_size[0]
                #
                # RoI_x = int(jsonObj["motion_RoI"]['x'] // scaleW)
                # RoI_y = int(jsonObj["motion_RoI"]['y'] // scaleH)
                # RoI_W = int(jsonObj["motion_RoI"]['width'] // scaleW)
                # RoI_H = int(jsonObj["motion_RoI"]['height'] // scaleH)
                #
                # print('RoI: %f, %f, %f, %f' % (RoI_x, RoI_y, RoI_W, RoI_H))
                #
                # estimated_roi = estimated[RoI_y:RoI_y + RoI_H, RoI_x:RoI_x + RoI_W, :]
                # gt_roi = gt[RoI_y:RoI_y + RoI_H, RoI_x:RoI_x + RoI_W, :]
                #
                # # whole image value
                #
                # this_ie = np.mean(np.sqrt(np.sum((estimated * 255 - gt * 255) ** 2, axis=2)))
                #
                # psnrs[validationIndex][tt] = this_psnr
                # # ssims[validationIndex][tt] = this_ssim
                # ies[validationIndex][tt] = this_ie
                #
                # psnr_whole += this_psnr
                # # ssim_whole += this_ssim
                # ie_whole += this_ie
                # outputs = None
                #
                # # value for difficulty levels
                # psnrs_level[diff[level]] += this_psnr
                # # ssims_level[diff[level]] += this_ssim
                # num_level[diff[level]] += 1
                #
                # # roi image value
                # this_roi_psnr = peak_signal_noise_ratio(estimated_roi, gt_roi)
                # # this_roi_ssim = structural_similarity(estimated_roi, gt_roi, multichannel=True, gaussian=True)
                #
                # psnr_roi += this_roi_psnr
                # # ssim_roi += this_roi_ssim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)

    inference(config)

