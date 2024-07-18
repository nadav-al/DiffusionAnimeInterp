import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as TF
import numpy as np
import sys
import os
import argparse

from .rfr_model.rfr_new import RFR as RFR
from .softsplat import ModuleSoftsplat as ForwardWarp
from .GridNet import GridNet

from utils.image_processing import generate_caption

from diffusers import ControlNetModel, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers import StableDiffusionImg2ImgPipeline



class FeatureExtractor(nn.Module):
    """The quadratic model"""
    def __init__(self, path='./network-default.pytorch'):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(96, 96, 3, padding=1)
        self.prelu6 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x1 = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x1))
        x2 = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x2))
        x3 = self.prelu6(self.conv6(x))

        return x1, x2, x3


class DiffimeInterp(nn.Module):
    """The quadratic model"""
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, init_diff=True, args=None):
        super(DiffimeInterp, self).__init__()

        args = argparse.Namespace()
        args.small = False
        args.mixed_precision = False
        # args.requires_sq_flow = False

        self.flownet = RFR(args)
        self.feat_ext = FeatureExtractor()
        self.fwarp = ForwardWarp('summation')
        self.synnet = GridNet(6, 64, 128, 96*2, 3)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.flownet = nn.DataParallel(self.flownet)
            self.feat_ext = nn.DataParallel(self.feat_ext)
            self.fwarp = nn.DataParallel(self.fwarp)
            self.synnet = nn.DataParallel(self.synnet)
        else:
            self.device = torch.device("cpu")

        if config.seed is not None:
            self.generator = torch.manual_seed(config.seed)
        else:
            self.generator = None

        normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
        normalize2 = TF.Normalize([0, 0, 0], config.std)
        self.trans = TF.Compose([TF.ToTensor(), normalize1, normalize2,])

        revmean = [-x for x in config.mean]
        revstd = [1.0 / x for x in config.std]
        revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
        revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
        self.revNormalize = TF.Compose([revnormalize1, revnormalize2])
        self.revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])
        self.to_img = TF.ToPILImage()

        self.config = config
        if init_diff:
            self.load_diffuser()
        # self.store_path = config.store_path
        self.counter = 0

        if path is not None:
            dict1 = torch.load(path)
            dict2 = dict()
            for key in dict1:
                dict2[key[7:]] = dict1[key]
            self.flownet.load_state_dict(dict2, strict=False)

    def load_diffuser(self, type="image", controlnet=None):
        print("loading diffuser")
        if controlnet is not None:
            if type == "image":
                self.pipeline = AutoPipelineForImage2Image.from_pretrained(self.config.diff_path,
                                                                           controlnet=controlnet,
                                                                           torch_dtype=torch.bfloat16, variant="fp16",
                                                                           use_safetensors=True).to("cuda")
            elif type == "text":
                self.pipeline = AutoPipelineForText2Image.from_pretrained(self.config.diff_path,
                                                                          controlnet=controlnet,
                                                                          torch_dtype=torch.bfloat16, variant="fp16",
                                                                          use_safetensors=True).to("cuda")
        else:
            if type == "image":
                self.pipeline = AutoPipelineForImage2Image.from_pretrained(self.config.diff_path,
                                                                           torch_dtype=torch.bfloat16, variant="fp16",
                                                                           use_safetensors=True).to("cuda")
            elif type == "text":
                self.pipeline = AutoPipelineForText2Image.from_pretrained(self.config.diff_path,
                                                                          torch_dtype=torch.bfloat16, variant="fp16",
                                                                          use_safetensors=True).to("cuda")
        print("diffuser loaded")
        self.pipeline.enable_model_cpu_offload()

    def dflow(self, flo, target):
        tmp = F.interpolate(flo, target.size()[2:4])
        tmp[:, :1] = tmp[:, :1].clone() * tmp.size()[3] / flo.size()[3]
        tmp[:, 1:] = tmp[:, 1:].clone() * tmp.size()[2] / flo.size()[2]

        return tmp

    def extract_features_2_frames(self, I1, I2):
        I1o = (I1 - 0.5) / 0.5
        I2o = (I2 - 0.5) / 0.5

        feat11, feat12, feat13 = self.feat_ext(I1o)
        feat21, feat22, feat23 = self.feat_ext(I2o)

        return I1o, [feat11, feat12, feat13], I2o, [feat21, feat22, feat23]

    def motion_calculation(self, Is, Ie, Flow, features, t, ind):
        """
        Args:
            Is: source image
            Ie: target image
            Flow: initial flow

            t: interpolation factor
            ind: index of the frame
        """
        F12, F12in, _ = self.flownet(Is, Ie, iters=12, test_mode=False, flow_init=Flow)
        if ind == 0:  # First frame
            Ft = t * F12
        else:  # Last frame
            Ft = (1 - t) * F12

        Ftd = self.dflow(Ft, features[0])
        Ftdd = self.dflow(Ft, features[1])
        Ftddd = self.dflow(Ft, features[2])

        return F12, F12in, [Ft.cuda(), Ftd.cuda(), Ftdd.cuda(), Ftddd.cuda()]


    def warping(self, Fts, I, features):
        Ft, Ftd, Ftdd, Ftddd = Fts
        feat1, feat2, feat3 = features

        one0 = torch.ones(I.size(), requires_grad=True).cuda()
        one1 = torch.ones(feat1.size(), requires_grad=True).cuda()
        one2 = torch.ones(feat2.size(), requires_grad=True).cuda()
        one3 = torch.ones(feat3.size(), requires_grad=True).cuda()

        I = I.cuda()


        It = self.fwarp(I, Ft)
        feat_t1 = self.fwarp(feat1, Ftd)
        feat_t2 = self.fwarp(feat2, Ftdd)
        feat_t3 = self.fwarp(feat3, Ftddd)

        norm = self.fwarp(one0, Ft.clone())
        norm_t1 = self.fwarp(one1, Ftd.clone())
        norm_t2 = self.fwarp(one2, Ftdd.clone())
        norm_t3 = self.fwarp(one3, Ftddd.clone())

        return It, [feat_t1, feat_t2, feat_t3], norm, [norm_t1, norm_t2, norm_t3]

    def normalize(self, It, feat_t, norm, norm_t):
        It[norm > 0] = It.clone()[norm > 0] / norm[norm > 0]
        feat_t[0][norm_t[0] > 0] = feat_t[0].clone()[norm_t[0] > 0] / norm_t[0][norm_t[0] > 0]
        feat_t[1][norm_t[1] > 0] = feat_t[1].clone()[norm_t[1] > 0] / norm_t[1][norm_t[1] > 0]
        feat_t[2][norm_t[2] > 0] = feat_t[2].clone()[norm_t[2] > 0] / norm_t[2][norm_t[2] > 0]

    def diffuse_latents(self, I1t, I2t, feat1t, feat2t, folder, style):
        # I1t_im = self.revtrans(I1t.cpu()[0])
        I1t_im = self.to_img(self.revNormalize(I1t.cpu()[0]).clamp(0.0, 1.0))
        # I2t_im = self.revtrans(I2t.cpu()[0])
        I2t_im = self.to_img(self.revNormalize(I2t.cpu()[0]).clamp(0.0, 1.0))
        # I1t_im = I1t_im.resize((512, 512))
        # I2t_im = I2t_im.resize((512, 512))
        caption1 = generate_caption(I1t_im, max_words=2, style=style)
        dI1t = self.pipeline(caption1,
                             width=I1t_im.width, height=I1t_im.height,
                             negative_prompt="Distorted. Black Spots. Bad Quality. ",
                             num_inferece_steps=30, image=I1t_im, strength=0.5).images[0]
        dI2t = self.pipeline(caption1,
                             width=I2t_im.width, height=I2t_im.height,
                             negative_prompt="Distorted. Black Spots. Bad Quality. ",
                             num_inferece_steps=30, image=I2t_im, strength=0.5).images[0]

        # resize
        dI1t = dI1t.resize(self.config.test_size)
        dI2t = dI2t.resize(self.config.test_size)


        path = self.config.store_path + '/' + folder + '/latents'
        if not os.path.exists(path):
            os.makedirs(path)
        I1t_im.save(path + '/I1t.png')
        I2t_im.save(path + '/I2t.png')
        dI1t.save(path + '/dI1t.png')
        dI2t.save(path + '/dI2t.png')
        self.counter += 1
        dI1t = self.trans(dI1t.convert('RGB')).to(self.device).unsqueeze(0)
        dI2t = self.trans(dI2t.convert('RGB')).to(self.device).unsqueeze(0)
        # synthesis
        return self.synnet(torch.cat([dI1t, dI2t], dim=1), torch.cat([feat1t[0], feat2t[0]], dim=1),
                              torch.cat([feat1t[1], feat2t[1]], dim=1), torch.cat([feat1t[2], feat2t[2]], dim=1))




    def forward(self, I1, I2, F12i, F21i, t, folder=None):
        # extract features

        I1o, features1, I2o, features2 = self.extract_features_2_frames(I1, I2)
        feat11, feat12, feat13 = features1
        feat21, feat22, feat23 = features2

        # calculate motion
        F12, F12in, F1ts = self.motion_calculation(I1o, I2o, F12i, [feat11, feat12, feat13], t, 0)
        F21, F21in, F2ts = self.motion_calculation(I2o, I1o, F21i, [feat21, feat22, feat23], t, 1)

        # warping 
        I1t, feat1t, norm1, norm1t = self.warping(F1ts, I1, features1)
        I2t, feat2t, norm2, norm2t = self.warping(F2ts, I2, features2)

        # normalize
        # Note: normalize in this way benefit training than the original "linear"
        self.normalize(I1t, feat1t, norm1, norm1t)
        self.normalize(I2t, feat2t, norm2, norm2t)

        if folder.startswith("Disney"):
            style = "Disney"
        else:
            style = "Anime"

        # diffusion
        if self.config.diff_objective == "latents":
            It_warp = self.diffuse_latents(I1t, I2t, feat1t, feat2t, folder, style)

        elif self.config.diff_objective == "result":
            # synthesis
            It_warp = self.synnet(torch.cat([I1t, I2t], dim=1), torch.cat([feat1t[0], feat2t[0]], dim=1),
                                  torch.cat([feat1t[1], feat2t[1]], dim=1),
                                  torch.cat([feat1t[2], feat2t[2]], dim=1))
            It_warp = self.to_img(self.revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))
            caption = generate_caption(It_warp, max_words=2, style=style)
            It_warp = self.pipeline(caption,
                                    negative_prompt="Blur. Bad Quality. Distorted. smudges.",
                                    num_inferece_steps=60, image=It_warp).images[0]
            It_warp = It_warp.resize(self.config.test_size)
            It_warp = self.trans(It_warp.convert('RGB')).to(self.device).unsqueeze(0)

        elif self.config.diff_objective == "both":
            It_warp = self.diffuse_latents(I1t, I2t, feat1t, feat2t, folder, style)
            It_warp = self.to_img(self.revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))
            # It_warp = It_warp.resize((512,512))
            caption = generate_caption(It_warp, max_words=2, style=style)
            It_warp = self.pipeline(caption,
                                    negative_prompt="Blur. Bad Quality. Distorted. smudges.",
                                    num_inferece_steps=60, image=It_warp).images[0]
            It_warp = It_warp.resize(self.config.test_size)
            It_warp = self.trans(It_warp.convert('RGB')).to(self.device).unsqueeze(0)

        else:
            It_warp = self.synnet(torch.cat([I1t, I2t], dim=1), torch.cat([feat1t[0], feat2t[0]], dim=1),
                                  torch.cat([feat1t[1], feat2t[1]], dim=1),
                                  torch.cat([feat1t[2], feat2t[2]], dim=1))
        return It_warp, F12, F21, F12in, F21in







