import models
import datas
import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from math import log10
import numpy as np
import pandas as pd
import datetime
from utils.config import Config
from utils.files_and_folders import generate_folder
import sys
import cv2
from utils.vis_flow import flow_to_color
import json
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-p',
                        action="store_true",
                        help="If training lora, use this flag to apply preprocess on the data")
    parser.add_argument('--train_lora',
                        action="store_true")
    parser.add_argument("--lora_data_path",
                        type=str,
                        default=None)
    parser.add_argument("--lora_weights_path",
                        type=str,
                        default=None)
    parser.add_argument("--ckpt",
                        type=int,
                        default=None)
    parser.add_argument("--multiscene",
                        action="store_true")
    parser.add_argument("--captioning",
                        action="store_true")
    parser.add_argument(
        "--rank",
        type=int,
        default=6,
        help="The rank of the LoRA",
    )
    parser.add_argument(
        "--data_repeats",
        type=int,
        default=1,
    )
    args, _ = parser.parse_known_args()
    return args


def save_flow_to_img(flow, des):
        f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
        fcopy = f.copy()
        fcopy[:, :, 0] = f[:, :, 1]
        fcopy[:, :, 1] = f[:, :, 0]
        cf = flow_to_color(-fcopy)
        cv2.imwrite(des + '.jpg', cf)


def lora_training(config, args, validationloader, model):
    if not args.train_lora:
        return args.lora_weights_path, len(os.listdir(args.lora_weights_path))
    data_path = config.testset_root
    if args.lora_data_path:
        data_path = args.lora_data_path
    folders = data_path.split('/')
    folder_base = folders[-1]
    if folder_base in ["frames", "lora"]:
        folder_base = folders[-2]
    test_details = {"ms": args.multiscene, "rk": args.rank, "prep": args.p, "drpt": args.data_repeats,
                    "cap": args.captioning}

    if args.multiscene:
        if args.lora_data_path is not None:

            weights_path = model.module.train_lora_single_scene(args.lora_data_path, folder_base=folder_base, test_details=test_details, weights_path=args.lora_weights_path, apply_preprocess=args.p,  with_caption=args.captioning)
        # elif args.lora_weights_path is not None and os.path.exists(args.lora_weights_path):
        #     weights_path = args.lora_weights_path
        else:
            weights_path = model.module.train_lora_multi_scene(config.testset_root, folder_base=folder_base, test_details=test_details, weights_path=args.lora_weights_path, apply_preprocess=args.p, with_caption=args.captioning)
        return weights_path, len(os.listdir(weights_path))
    else:
        weights_paths = {}
        ckpt_amount = 0
        for validationIndex, validationData in enumerate(validationloader, 0):
            sample, flow, index, folder = validationData
            if args.lora_weights_path is not None and \
                    os.path.exists(os.path.join(args.lora_weights_path, folder[0][0])):
                weights_paths[folder[0][0]] = (os.path.join(args.lora_weights_path, folder[0][0]))
            else:
                lora_folder = os.path.join(config.testset_root, folder[0][0])
                weights_paths[folder[0][0]] = (model.module.train_lora_single_scene(lora_folder, folder_name=folder[0][0], folder_base=folder_base, test_details=test_details, weights_path=args.lora_weights_path, apply_preprocess=args.p, with_caption=args.captioning))
            if validationIndex == 0:
                ckpt_amount = len(os.listdir(weights_paths[folder[0][0]]))
        return weights_paths, ckpt_amount


def validate_on_lora(config, args):
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

    print(testset)
    sys.stdout.flush()

    device = torch.device("cuda")

    # prepare model
    model = getattr(models, config.model)(config.pwc_path, config=config).to(device)
    model = nn.DataParallel(model)
    retImg = []

    # load weights
    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'], strict=False)

    # prepare others
    store_path = config.store_path
    weights_path, ckpt_amount = lora_training(config, args, validationloader, model)



    ## values for whole image
    psnr_whole = 0
    # psnrs = np.zeros([ckpt_amount, len(testset), config.inter_frames])
    ie_whole = 0
    # ies = np.zeros([ckpt_amount, len(testset), config.inter_frames])

    columns = ['ValidIndex'] + [f'Frame_{i+2}' for i in range(config.inter_frames)]
    psnr_df = pd.DataFrame(columns=columns)
    psnr_rows = []
    ie_df = pd.DataFrame(columns=columns)
    ie_rows = []

    ckpt = "pytorch_lora_weights.safetensors"
    if args.ckpt != -1:
        ckpt = os.path.join(f"checkpoint-{args.ckpt}", ckpt)

    folders = []

    print('Everything prepared. Ready to test...')
    sys.stdout.flush()

    #  start testing...
    with torch.no_grad():
        model.eval()
        ii = 0

        for validationIndex, validationData in enumerate(validationloader, 0):
            sys.stdout.flush()
            sample, flow, index, folder = validationData

            first_frame = sample[0]
            last_frame = sample[-1]

            folders.append(folder[0][0])

            # initial SGM flow
            F12i, F21i = flow

            if torch.cuda.is_available():
                F12i = F12i.float().cuda()
                F21i = F21i.float().cuda()
            else:
                F12i = F12i.float()
                F21i = F21i.float()

            ITs = [sample[tt] for tt in range(1, 2)]

            if torch.cuda.is_available():
                I1 = first_frame.cuda()
                I2 = last_frame.cuda()
            else:
                I1 = first_frame
                I2 = last_frame

            output_path = generate_folder(folder[0][0], folder_base="", root_path=config.store_path, test_details=test_details, unique_folder=(validationIndex == 0))
            flow_output_path = os.path.join(output_path, 'flows')

            if not os.path.exists(flow_output_path):
                os.makedirs(flow_output_path)

            num_frames = config.inter_frames
            revtrans(I1.cpu()[0]).save(os.path.join(output_path, 'frame1.png'))
            revtrans(I2.cpu()[0]).save(os.path.join(output_path, f'frame{num_frames + 2}.png'))
            x = config.inter_frames
            if args.multiscene:
                ckpt_root = weights_path
            else:
                ckpt_root = weights_path[folder[0][0]]

            ckpt_path = os.path.join(ckpt_root, ckpt)

            psnrs = []
            ies = []

            for tt in range(config.inter_frames):
                t = 1.0 / (x + 1) * (tt + 1)

                if torch.cuda.is_available():
                    outputs = model.module(I1, I2, F12i, F21i, t, folder=folder[0][0], weights_path=ckpt_path)
                else:
                    outputs = model(I1, I2, F12i, F21i, t, folder=folder[0][0], weights_path=ckpt_path)


                It_warp = revNormalize(outputs[0].cpu()[0]).clamp(0.0, 1.0)
                It_gt = revNormalize(ITs[tt][0]).clamp(0.0, 1.0)

                warp_img = to_img(It_warp)
                warp_img.save(os.path.join(output_path, f'frame{tt + 2}.png'))
                gt_img = to_img(It_gt)
                gt_img.save(os.path.join(output_path, f'gt_frame{tt + 2}.png'))

                if tt == 0:
                    save_flow_to_img(outputs[1].cpu(), os.path.join(flow_output_path, 'F12'))
                    save_flow_to_img(outputs[2].cpu(), os.path.join(flow_output_path, 'F21'))

                estimated = It_warp.numpy().transpose(1, 2, 0)
                gt = It_gt.numpy().transpose(1, 2, 0)

                # whole image value
                this_psnr = peak_signal_noise_ratio(estimated, gt)
                this_ie = np.mean(np.sqrt(np.sum((estimated * 255 - gt * 255) ** 2, axis=2)))

                psnrs.append(this_psnr)
                ies.append(this_ie)

                psnr_whole += this_psnr
                ie_whole += this_ie


            psnr_row = {'ValidIndex': validationIndex}
            psnr_row.update({f"Frame_{tt+2}": psnr for tt, psnr in enumerate(psnrs)})
            psnr_rows.append(psnr_row)
            # psnr_df._append(psnr_row, ignore_index=True)

            ie_row = {'ValidIndex': validationIndex}
            ie_row.update({f"Frame_{tt+2}": ie for tt, ie in enumerate(ies)})
            ie_rows.append(ie_row)
            # ie_df._append(ie_row, ignore_index=True)

        # psnrs_whole = psnr_df.groupby('ckpt').
        psnr_df = pd.DataFrame(psnr_rows)
        ie_df = pd.DataFrame(ie_rows)
        psnrs_mean = psnr_df.mean()
        ies_mean = ie_df.mean()

    output_path = generate_folder(folder_base="", root_path=config.store_path, test_details=test_details)
    return psnr_df, ie_df, output_path, weights_path




def validate_checkpoints_on_lora(config, args):
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

    print(testset)
    sys.stdout.flush()

    device = torch.device("cuda")

    # prepare model
    model = getattr(models, config.model)(config.pwc_path, config=config).to(device)
    model = nn.DataParallel(model)
    retImg = []

    # load weights
    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'], strict=False)


    # prepare others
    store_path = config.store_path
    weights_path, ckpt_amount = lora_training(config, args, validationloader, model)



    ## values for whole image
    psnr_whole = 0
    # psnrs = np.zeros([ckpt_amount, len(testset), config.inter_frames])
    ie_whole = 0
    # ies = np.zeros([ckpt_amount, len(testset), config.inter_frames])

    columns = ['ckpt', 'ValidIndex'] + [f'Frame_{i+2}' for i in range(config.inter_frames)]
    psnr_df = pd.DataFrame(columns=columns)
    psnr_rows = []
    ie_df = pd.DataFrame(columns=columns)
    ie_rows = []

    folders = []

    print('Everything prepared. Ready to test...')
    sys.stdout.flush()

    #  start testing...
    with torch.no_grad():
        model.eval()
        ii = 0

        for validationIndex, validationData in enumerate(validationloader, 0):
            sys.stdout.flush()
            sample, flow, index, folder = validationData

            first_frame = sample[0]
            last_frame = sample[-1]

            folders.append(folder[0][0])

            # initial SGM flow
            F12i, F21i = flow

            if torch.cuda.is_available():
                F12i = F12i.float().cuda()
                F21i = F21i.float().cuda()
            else:
                F12i = F12i.float()
                F21i = F21i.float()

            ITs = [sample[tt] for tt in range(1, 2)]

            if torch.cuda.is_available():
                I1 = first_frame.cuda()
                I2 = last_frame.cuda()
            else:
                I1 = first_frame
                I2 = last_frame

            output_path = generate_folder(folder[0][0], folder_base="", root_path=config.store_path, test_details=test_details, unique_folder=(validationIndex == 0))
            flow_output_path = os.path.join(output_path, 'flows')

            if not os.path.exists(flow_output_path):
                os.makedirs(flow_output_path)

            num_frames = config.inter_frames
            revtrans(I1.cpu()[0]).save(os.path.join(output_path, 'frame1.png'))
            revtrans(I2.cpu()[0]).save(os.path.join(output_path, f'frame{num_frames + 2}.png'))
            x = config.inter_frames
            if args.multiscene:
                ckpt_root = weights_path
            else:
                ckpt_root = weights_path[folder[0][0]]

            checkpoints = sorted(os.listdir(ckpt_root))
            for i, ckpt in enumerate(checkpoints):
                ckpt_path = os.path.join(ckpt_root, ckpt)
                if os.path.isfile(ckpt_path):
                    ckpt_path = ckpt_root
                    ckpt_id = ckpt_amount
                else:
                    ckpt_id = int(ckpt.replace("checkpoint-", ""))

                psnrs = []
                ies = []

                for tt in range(config.inter_frames):
                    t = 1.0 / (x + 1) * (tt + 1)

                    if torch.cuda.is_available():
                        outputs = model.module(I1, I2, F12i, F21i, t, folder=folder[0][0], weights_path=ckpt_path)

                    else:
                        outputs = model(I1, I2, F12i, F21i, t, folder=folder[0][0], weights_path=ckpt_path)


                    It_warp = outputs[0]

                    It_warp = revNormalize(outputs[0].cpu()[0]).clamp(0.0, 1.0)
                    It_gt = revNormalize(ITs[tt][0]).clamp(0.0, 1.0)

                    warp_img = to_img(It_warp)
                    warp_img.save(os.path.join(output_path, f'frame{tt + 2}.png'))
                    gt_img = to_img(It_gt)
                    gt_img.save(os.path.join(output_path, f'gt_frame{tt + 2}.png'))

                    warp_img = to_img(warp_img)
                    warp_img.save(os.path.join(output_path, f'ckpt{ckpt_id}_frame{tt + 2}.png'))

                    if tt == 0:
                        save_flow_to_img(outputs[1].cpu(), os.path.join(flow_output_path, 'F12'))
                        save_flow_to_img(outputs[2].cpu(), os.path.join(flow_output_path, 'F21'))

                    estimated = revNormalize(It_warp[0].cpu()).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
                    gt = revNormalize(ITs[tt][0]).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)

                    # whole image value
                    this_psnr = peak_signal_noise_ratio(estimated, gt)
                    this_ie = np.mean(np.sqrt(np.sum((estimated * 255 - gt * 255) ** 2, axis=2)))

                    psnrs.append(this_psnr)
                    ies.append(this_ie)

                    psnr_whole += this_psnr
                    ie_whole += this_ie


                psnr_row = {'ckpt': ckpt_id, 'ValidIndex': validationIndex}
                psnr_row.update({f"Frame_{tt+2}": psnr for tt, psnr in enumerate(psnrs)})
                psnr_rows.append(psnr_row)
                # psnr_df._append(psnr_row, ignore_index=True)

                ie_row = {'ckpt': ckpt_id, 'ValidIndex': validationIndex    }
                ie_row.update({f"Frame_{tt+2}": ie for tt, ie in enumerate(ies)})
                ie_rows.append(ie_row)
                # ie_df._append(ie_row, ignore_index=True)

        # psnrs_whole = psnr_df.groupby('ckpt').
        psnr_df = pd.DataFrame(psnr_rows)
        ie_df = pd.DataFrame(ie_rows)
        psnrs_mean = psnr_df.groupby(['ckpt']).mean()
        ies_mean = ie_df.groupby(['ckpt']).mean()

    output_path = generate_folder(folder_base="", root_path=config.store_path, test_details=test_details)
    return psnr_df, ie_df, output_path, weights_path

# TODO GIANT!!!!
#   Finish creating the saving of the losses, maybe change to pandas
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

    revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

    testset = datas.AniTripletWithSGMFlowTest(config.testset_root, config.test_flow_root, trans, config.test_size, config.test_crop_size, train=False)
    sampler = torch.utils.data.SequentialSampler(testset)
    validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)
    to_img = TF.ToPILImage()
 
    print(testset)
    sys.stdout.flush()

    device = torch.device("cuda")

    # prepare model
    # if config.model in ['AnimeInterp', 'AnimeInterpNoCupy']:
    #     model = getattr(models, config.model)(config.pwc_path).to(device)
    # else:
    #     model = getattr(models, config.model)(config.pwc_path, config=config).to(device)
    model = getattr(models, config.model)(config.pwc_path, config=config).to(device)
    model = nn.DataParallel(model)
    retImg = []


    # load weights
    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'], strict=False)

    # prepare others
    store_path = config.store_path

    ## values for whole image
    psnr_whole = 0
    # psnrs = np.zeros([len(testset), config.inter_frames])
    psrns = []
    ie_whole = 0
    # ies = np.zeros([len(testset), config.inter_frames])

    columns = ['ValidIndex'] + [f'Frame_{i + 2}' for i in range(config.inter_frames)]
    psnr_df = pd.DataFrame(columns=columns)
    psnr_rows = []
    ie_df = pd.DataFrame(columns=columns)
    ie_rows = []


    folders = []

    print('Everything prepared. Ready to test...')  
    sys.stdout.flush()

    #  start testing...
    with torch.no_grad():
        model.eval()
        ii = 0
        for validationIndex, validationData in enumerate(validationloader, 0):
            sys.stdout.flush()
            sample, flow, index, folder = validationData

            first_frame = sample[0]
            last_frame = sample[-1]

            folders.append(folder[0][0])

            # initial SGM flow
            F12i, F21i  = flow

            if torch.cuda.is_available():
                F12i = F12i.float().cuda()
                F21i = F21i.float().cuda()
            else:
                F12i = F12i.float()
                F21i = F21i.float()

            ITs = [sample[tt] for tt in range(1, 2)]

            if torch.cuda.is_available():
                I1 = first_frame.cuda()
                I2 = last_frame.cuda()
            else:
                I1 = first_frame
                I2 = last_frame

            output_path = generate_folder(folder[0][0], folder_base="", root_path=config.store_path, unique_folder=(validationIndex == 0))
            flow_output_path = os.path.join(output_path, 'flows')

            if not os.path.exists(flow_output_path):
                os.makedirs(flow_output_path)

            psnrs = []
            ies = []

            num_frames = config.inter_frames
            revtrans(I1.cpu()[0]).save(os.path.join(output_path, 'frame1.png'))
            revtrans(I2.cpu()[0]).save(os.path.join(output_path, f'frame{num_frames+2}.png'))
            x = config.inter_frames
            for tt in range(config.inter_frames):
                t = 1.0/(x+1) * (tt + 1)

                if torch.cuda.is_available():
                    outputs = model.module(I1, I2, F12i, F21i, t, folder=folder[0][0])
                else:
                    outputs = model(I1, I2, F12i, F21i, t, folder=folder[0][0])

                It_warp = outputs[0]
                warp_img = to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))

                warp_img.save(os.path.join(output_path, f'frame{tt + 2}.png'))
                if tt == 0:
                    save_flow_to_img(outputs[1].cpu(), os.path.join(flow_output_path, 'F12'))
                    save_flow_to_img(outputs[2].cpu(), os.path.join(flow_output_path, 'F21'))

                estimated = revNormalize(It_warp[0].cpu()).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
                gt = revNormalize(ITs[tt][0]).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)


                # whole image value
                this_psnr = peak_signal_noise_ratio(estimated, gt)
                this_ie = np.mean(np.sqrt(np.sum((estimated*255 - gt*255)**2, axis=2)))

                psnrs.append(this_psnr)
                ies.append(this_ie)

                psnr_whole += this_psnr
                ie_whole += this_ie

            psnr_row = {'ValidIndex': validationIndex}
            psnr_row.update({f"Frame_{tt + 2}": psnr for tt, psnr in enumerate(psnrs)})
            psnr_rows.append(psnr_row)

            ie_row = {'ValidIndex': validationIndex}
            ie_row.update({f"Frame_{tt + 2}": ie for tt, ie in enumerate(ies)})
            ie_rows.append(ie_row)

        psnr_df = pd.DataFrame(psnr_rows)
        ie_df = pd.DataFrame(ie_rows)


    return psnr_df, ie_df, folders, output_path




if __name__ == "__main__":
    # loading configures
    args = parse_args()
    config = Config.from_file(args.config)

    root = "experiments/outputs"
    if args.lora_weights_path:
        folders = args.lora_weights_path.split('/')
        if folders[-1].startswith("test"):
            test_details = folders[-1]
        else:
            test_details = folders[-2]



    if config.model in [ 'LoraInterp', 'LoraCNInterp' ]:
        if args.ckpt is not None:
            config.store_path = os.path.join(config.store_path, str(args.ckpt))
            psnrs, ies, output_path, weights_path = validate_on_lora(config, args)
        else:
            psnrs, ies, output_path, weights_path = validate_checkpoints_on_lora(config, args)
    else:
        psnrs, ies, folder, output_path = validate(config, args)

    psnrs.to_csv(os.path.join(output_path, "psnr.csv"))
    ies.to_csv(os.path.join(output_path, "ie.csv"))

