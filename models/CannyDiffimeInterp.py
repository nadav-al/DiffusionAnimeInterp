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
from .DiffimeInterp import DiffimeInterp

from diffusers import ControlNetModel, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers import StableDiffusionImg2ImgPipeline

import cv2


def _tensor_to_canny(tensor):
    """
      Converts a PyTorch tensor of a batch of images to cv2 Canny edge images while maintaining the batch structure.

      Args:
          tensor: A PyTorch tensor of shape (B, C, H, W) representing a batch of images.

      Returns:
          A PyTorch tensor of shape (B, C, H, W) containing the Canny edge images for each input image.
      """
    # Move tensor to CPU if on GPU
    if tensor.device.type == "cuda":
        tensor = tensor.cpu()

    # Create an empty tensor to store Canny edge images
    canny_images = torch.empty_like(tensor)

    # Loop through each image in the batch and apply Canny edge detection
    for i in range(tensor.shape[0]):
        image_tensor = tensor[i]
        canny_image = _process_single_image(image_tensor)
        canny_images[i] = torch.from_numpy(canny_image)  # Convert NumPy array to PyTorch tensor

    return canny_images

def _process_single_image(image_tensor, canny_th_low=50, canny_th_high=150):
    # Convert tensor to numpy array and swap color channels (CHW -> HWC)
    np_img = image_tensor.permute(1, 2, 0).numpy()


    # Handle possible normalization if the tensor values are between 0 and 1
    if np_img.dtype == np.float32:
        np_img *= 255.0

    # Convert to uint8 for cv2 compatibility
    np_img = np_img.astype(np.uint8)

    np_grayscale = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(np_grayscale, canny_th_low, canny_th_high)


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


class CannyDiffimeInterp(DiffimeInterp):
    """The quadratic model"""
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, canny_th_low=50, canny_th_high=150, args=None):
        super(CannyDiffimeInterp, self).__init__(path, config, False, args)
        self.cth_low = canny_th_low
        self.cth_heigh = canny_th_high

        print("loading controlnet")
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        print("controlnet loaded")

        self.load_diffuser(type="text", controlnet=controlnet)

        print("loading adapter")
        self.pipeline.load_ip_adapter(config.ip_adapter_id, subfolder='sdxl_models', weight_name='ip-adapter-plus_sdxl_vit-h.safetensors')
        self.pipeline.set_ip_adapter_scale(0.8)
        print("adapter loaded")


    def forward(self, I1, I2, F12i, F21i, t, folder=None):
        # extract features
        I1o, features1, I2o, features2 = self.extract_features_2_frames(I1, I2)
        feat11, feat12, feat13 = features1
        feat21, feat22, feat23 = features2

        # calculate motion
        F12, F12in, F1t = self.motion_calculation(I1o, I2o, F12i, [feat11, feat12, feat13], t, 0)
        F21, F21in, F2t = self.motion_calculation(I2o, I1o, F21i, [feat21, feat22, feat23], t, 1)


        # canny edge
        I1c = _tensor_to_canny(I1)
        I2c = _tensor_to_canny(I2)

        # warping
        w_I1c, feat1t, norm1, norm1t = self.warping(F1t, I1c, features1)
        w_I2c, feat2t, norm2, norm2t = self.warping(F2t, I2c, features2)


        w_I1c_img = self.revNormalize(w_I1c.cpu()[0]).unsqueeze(0)
        w_I2c_img = self.revNormalize(w_I2c.cpu()[0]).unsqueeze(0)

        print("canny created")
        # print(w_I2c_img)
        I1_im = self.revNormalize(I1.cpu()[0]).unsqueeze(0)
        I2_im = self.revNormalize(I2.cpu()[0]).unsqueeze(0)
        # diffuser
        d_I1c = self.pipeline("", image=w_I1c_img, ip_adapter_image=I1_im)
        d_I2c = self.pipeline("", image=w_I2c_img, ip_adapter_image=I2_im)

        print("diffused")

        # for exploration and understanding the model, saves the intermediate results
        # self.revtrans(w_I1c.cpu()[0]).save(f'{self.store_path}/canny_I1_{self.counter}.png')
        # self.revtrans(w_I2c.cpu()[0]).save(f'{self.store_path}/canny_I2_{self.counter}.png')
        d_I1c.save(f'{self.store_path}/cDiff_I1_{self.counter}.png')
        d_I2c.save(f'{self.store_path}/cDiff_I2_{self.counter}.png')
        self.counter += 1

        self.normalize(d_I1c, feat1t, norm1, norm1t)
        self.normalize(d_I2c, feat2t, norm2, norm2t)

        # synthesis
        It_warp = self.synnet(torch.cat([d_I1c, d_I2c], dim=1), torch.cat([feat1t[0], feat2t[0]], dim=1),
                              torch.cat([feat1t[1], feat2t[1]], dim=1), torch.cat([feat1t[2], feat2t[2]], dim=1))
        #
        # warp_im = TF.ToPILImage(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))

        return It_warp, F12, F21, F12in, F21in
