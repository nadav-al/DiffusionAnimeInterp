import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as TF
import numpy as np
import sys
import argparse

from .rfr_model.rfr_new import RFR as RFR
from .softsplat import ModuleSoftsplat as ForwardWarp
from .GridNet import GridNet

from diffusers import ControlNetModel, AutoPipelineForText2Image, AutoPipelineForImage2Image



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
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, base_diff=None, args=None):
        super(DiffimeInterp, self).__init__()

        args = argparse.Namespace()
        args.small = False
        args.mixed_precision = False
        # args.requires_sq_flow = False

        self.flownet = RFR(args)
        self.feat_ext = FeatureExtractor()
        self.fwarp = ForwardWarp('summation')
        self.synnet = GridNet(6, 64, 128, 96*2, 3)

        normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
        normalize2 = TF.Normalize([0, 0, 0], config.std)
        self.trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

        revmean = [-x for x in config.mean]
        revstd = [1.0 / x for x in config.std]
        revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
        revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
        self.revNormalize = TF.Compose([revnormalize1, revnormalize2])
        self.revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

        self.diff_objective = config.diff_objective
        print("loading diffuser")
        if base_diff is not None:
            self.pipline = AutoPipelineForImage2Image(**base_diff.components).to("cuda")
            pass
        else:
            self.pipline = AutoPipelineForImage2Image.from_pretrained(config.diff_path, torch_dtype=torch.float16,
                                                                      variant="fp16", use_safetensors=True).to("cuda")
        print("diffuser loaded")
        self.pipline.enable_model_cpu_offload()

        self.store_path = config.store_path
        self.counter = 0

        if path is not None:
            dict1 = torch.load(path)
            dict2 = dict()
            for key in dict1:
                dict2[key[7:]] = dict1[key]
            self.flownet.load_state_dict(dict2, strict=False)

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

        return F12, F12in, Ft, Ftd, Ftdd, Ftddd

    def forward(self, I1, I2, F12i, F21i, t):
        # extract features
        I1o, features1, I2o, features2 = self.extract_features_2_frames(I1, I2)
        feat11, feat12, feat13 = features1
        feat21, feat22, feat23 = features2

        # calculate motion
        F12, F12in, F1t, F1td, F1tdd, F1tddd = self.motion_calculation(I1o, I2o, F12i, [feat11, feat12, feat13], t, 0)
        F21, F21in, F2t, F2td, F2tdd, F2tddd = self.motion_calculation(I2o, I1o, F21i, [feat21, feat22, feat23], t, 1)

        # warping 
        one0 = torch.ones(I1.size(), requires_grad=True).cuda()
        one1 = torch.ones(feat11.size(), requires_grad=True).cuda()
        one2 = torch.ones(feat12.size(), requires_grad=True).cuda()
        one3 = torch.ones(feat13.size(), requires_grad=True).cuda()

        I1t = self.fwarp(I1, F1t)
        feat1t1 = self.fwarp(feat11, F1td)
        feat1t2 = self.fwarp(feat12, F1tdd)
        feat1t3 = self.fwarp(feat13, F1tddd)

        I2t = self.fwarp(I2, F2t)
        feat2t1 = self.fwarp(feat21, F2td)
        feat2t2 = self.fwarp(feat22, F2tdd)
        feat2t3 = self.fwarp(feat23, F2tddd)

        norm1 = self.fwarp(one0, F1t.clone())
        norm1t1 = self.fwarp(one1, F1td.clone())
        norm1t2 = self.fwarp(one2, F1tdd.clone())
        norm1t3 = self.fwarp(one3, F1tddd.clone())

        norm2 = self.fwarp(one0, F2t.clone())
        norm2t1 = self.fwarp(one1, F2td.clone())
        norm2t2 = self.fwarp(one2, F2tdd.clone())
        norm2t3 = self.fwarp(one3, F2tddd.clone())

        # normalize
        # Note: normalize in this way benefit training than the original "linear"
        I1t[norm1 > 0] = I1t.clone()[norm1 > 0] / norm1[norm1 > 0]
        I2t[norm2 > 0] = I2t.clone()[norm2 > 0] / norm2[norm2 > 0]
        
        feat1t1[norm1t1 > 0] = feat1t1.clone()[norm1t1 > 0] / norm1t1[norm1t1 > 0]
        feat2t1[norm2t1 > 0] = feat2t1.clone()[norm2t1 > 0] / norm2t1[norm2t1 > 0]
        
        feat1t2[norm1t2 > 0] = feat1t2.clone()[norm1t2 > 0] / norm1t2[norm1t2 > 0]
        feat2t2[norm2t2 > 0] = feat2t2.clone()[norm2t2 > 0] / norm2t2[norm2t2 > 0]
        
        feat1t3[norm1t3 > 0] = feat1t3.clone()[norm1t3 > 0] / norm1t3[norm1t3 > 0]
        feat2t3[norm2t3 > 0] = feat2t3.clone()[norm2t3 > 0] / norm2t3[norm2t3 > 0]


        # diffusion
        if self.diff_objective == "latent":
            I1t_im = self.revtrans(I1t)
            I2t_im = self.revtrans(I2t)
            dI1t = self.pipline("High quality. 2D classic animation. Clean. ",
                                negative_prompt="Distorted. Black Spots. Bad Quality. ", image=I1t_im).images[0]
            dI2t = self.pipline("High quality. 2D classic animation. Clean. ",
                                negative_prompt="Distorted. Black Spots. Bad Quality. ", image=I2t_im).images[0]

            I1t_im.save(self.store_path + f'latents/fold{self.counter}/I1t.png')
            I2t_im.save(self.store_path + f'latents/fold{self.counter}/I2t.png')
            dI1t.save(self.store_path + f'latents/fold{self.counter}/dI1t.png')
            dI2t.save(self.store_path + f'latents/fold{self.counter}/dI2t.png')
            self.counter += 1

            dI1t = self.trans(dI1t)
            dI2t = self.trans(dI2t)
            # synthesis
            It_warp = self.synnet(torch.cat([dI1t, dI2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1), torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

        else:
            # synthesis
            It_warp = self.synnet(torch.cat([I1t, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
                                  torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

            It_warp = self.pipline("High quality. 2D classic animation. Clean. ", image=It_warp).images[0]


        return It_warp, F12, F21, F12in, F21in




