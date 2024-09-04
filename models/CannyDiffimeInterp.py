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
from utils.captionning import generate_caption, generate_keywords
from utils.files_and_folders import extract_style_name, generate_folder

from transformers import CLIPVisionModelWithProjection
from diffusers import ControlNetModel, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers import StableDiffusionImg2ImgPipeline

from PIL import Image

import cv2


def _tensor_to_canny(tensor, canny_th_low=100, canny_th_high=200):
    """
      Converts a PyTorch tensor of a batch of images to cv2 Canny edge images while maintaining the batch structure.

      Args:
          tensor: A PyTorch tensor of shape (B, C, H, W) representing a batch of images.

      Returns:
          A PyTorch tensor of shape (B, C, H, W) containing the Canny edge images for each input image.
      """
    # Move tensor to CPU if on GPU
    device = tensor.device.type
    if device == "cuda":
        tensor = tensor.cpu()

    # Create an empty tensor to store Canny edge images
    canny_images = torch.empty_like(tensor)

    # Loop through each image in the batch and apply Canny edge detection
    for i in range(tensor.shape[0]):
        image_tensor = tensor[i]
        canny_image = _process_single_image(image_tensor)
        canny_images[i] = torch.from_numpy(canny_image)  # Convert NumPy array to PyTorch tensor

    return canny_images.to(device)

def _process_single_image(image_tensor, canny_th_low=100, canny_th_high=200):
    # Convert tensor to numpy array and swap color channels (CHW -> HWC)
    np_img = image_tensor.permute(1, 2, 0).numpy()


    # Handle possible normalization if the tensor values are between 0 and 1
    if np_img.dtype == np.float32:
        np_img *= 255.0

    # Convert to uint8 for cv2 compatibility
    np_img = np_img.astype(np.uint8)

    np_grayscale = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(np_grayscale, canny_th_low, canny_th_high)

def create_canny(original_image, canny_th_low, canny_th_high):
    image = np.array(original_image)
    image = cv2.Canny(image, canny_th_low, canny_th_high)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


NIS = 50  # num_inference_steps


class CannyDiffimeInterp(DiffimeInterp):
    """The quadratic model"""
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, args=None, canny_th_low=50, canny_th_high=150):
        super(CannyDiffimeInterp, self).__init__(path, config, False, args)
        self.cth_low = canny_th_low
        self.cth_high = canny_th_high

        self.load_diffuser()
        if config.diff_objective == "both":
            self.pipeline2 = AutoPipelineForImage2Image.from_pretrained(self.config.diff_path,
                                                                        torch_dtype=torch.float16, variant="fp16",
                                                                        use_safetensors=True).to("cuda")

    def set_low_threshold(self, value):
        self.cth_low = value

    def set_high_threshold(self, value):
        self.cth_high = value


    def load_diffuser(self, type="image"):
        print("loading controlnet")
        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        if type == "image":
            self.pipeline = AutoPipelineForImage2Image.from_pretrained(self.config.diff_path,
                                                                       controlnet=controlnet,
                                                                       torch_dtype=torch.float16, variant="fp16",
                                                                       use_safetensors=True).to("cuda")
        elif type == "text":
            self.pipeline = AutoPipelineForText2Image.from_pretrained(self.config.diff_path,
                                                                      controlnet=controlnet,
                                                                      torch_dtype=torch.float16, variant="fp16",
                                                                      use_safetensors=True).to("cuda")
        print("controlnet loaded")
        self.pipeline.enable_model_cpu_offload()


    def forward(self, I1, I2, F12i, F21i, t, folder=None, test_details=""):
        store_latents_path = generate_folder("latents", folder_base=test_details, root_path=self.config.store_path, test_details=folder)
        # if not os.path.exists(store_latents_path):
        #     os.makedirs(store_latents_path)


        # extract features
        I1o, features1, I2o, features2 = self.extract_features_2_frames(I1, I2)

        # calculate motion
        F12, F12in, F1t = self.motion_calculation(I1o, I2o, F12i, features1, t, 0)
        F21, F21in, F2t = self.motion_calculation(I2o, I1o, F21i, features2, t, 1)
        # breakpoint()

        folder_path = os.path.join(self.config.testset_root, folder)
        folder_path = os.path.normpath(folder_path)
        images_path = os.listdir(folder_path)
        style = extract_style_name(folder)

        im1 = Image.open(os.path.join(folder_path, images_path[0])).resize(self.config.test_size)
        im2 = Image.open(os.path.join(folder_path, images_path[-1])).resize(self.config.test_size)

        # canny edge
        I1c = create_canny(im1, self.cth_low, self.cth_high)
        I1c = self.trans(I1c.convert('RGB')).to(self.device).unsqueeze(0)
        I2c = create_canny(im2, self.cth_low, self.cth_high)
        I2c = self.trans(I2c.convert('RGB')).to(self.device).unsqueeze(0)

        # warping
        w_I1c, feat1t, norm1, norm1t = self.warping(F1t, I1c, features1)
        w_I2c, feat2t, norm2, norm2t = self.warping(F2t, I2c, features2)

        w_I1c, feat1t = self.normalize(w_I1c, feat1t, norm1, norm1t)
        w_I2c, feat2t = self.normalize(w_I2c, feat2t, norm2, norm2t)

        w_I1c_img = self.to_img(self.revNormalize(w_I1c.cpu()[0]).clamp(0.0, 1.0))
        w_I1c_img.save(store_latents_path + "/I1_canny.png")
        w_I2c_img = self.to_img(self.revNormalize(w_I2c.cpu()[0]).clamp(0.0, 1.0))
        w_I2c_img.save(store_latents_path + "/I2_canny.png")
        # breakpoint()
        print("canny created")


        if test_details != "" and "cap" in test_details.split('_'):
            caption1 = generate_caption(im1, max_words=2, style=style)
            caption2 = generate_caption(im2, max_words=2, style=style)
        else:
            caption1 = generate_keywords(style, max_words=4)
            caption2 = generate_keywords(style, max_words=4)

        width, height = self.config.test_size

        d_I1c = self.pipeline(caption1,
                              negative_prompt="worst quality. blurry. indistinct facial features. motion blur. faded colors. abstract. unclear background. washed out. distorted.",
                              width=width, height=height,
                              image=im1, control_image=w_I1c_img,
                              num_inference_steps=NIS, strength=0.45
                              ).images[0].resize(self.config.test_size)
        d_I2c = self.pipeline(caption2,
                              negative_prompt="worst quality. blurry. indistinct facial features. motion blur. faded colors. abstract. unclear background. washed out. distorted.",
                              width=width, height=height,
                              image=im2, control_image=w_I2c_img,
                              num_inference_steps=NIS, strength=0.45
                              ).images[0].resize(self.config.test_size)
        print("diffused")

        print(store_latents_path)
        d_I1c.save(f'{store_latents_path}/lat_frame1.png')
        d_I2c.save(f'{store_latents_path}/lat_frame3.png')

        d_I1c = self.trans(d_I1c.convert('RGB')).to(self.device).unsqueeze(0)
        d_I2c = self.trans(d_I2c.convert('RGB')).to(self.device).unsqueeze(0)


        # d_I1c, feat1t = self.normalize(d_I1c, feat1t, norm1, norm1t)
        # d_I2c, feat2t = self.normalize(d_I2c, feat2t, norm2, norm2t)

        # synthesis
        It_warp = self.synnet(torch.cat([d_I1c, d_I2c], dim=1), torch.cat([feat1t[0], feat2t[0]], dim=1),
                              torch.cat([feat1t[1], feat2t[1]], dim=1), torch.cat([feat1t[2], feat2t[2]], dim=1))

        if self.config.diff_objective == "both":
            output_path = generate_folder(folder, folder_base="", root_path=self.config.store_path,
                                          test_details=test_details)
            warp_img = self.to_img(self.revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))
            warp_img.save(os.path.join(output_path, "AniInterp_frame2.png"))
            if test_details != "" and "cap" in test_details.split('_'):
                caption = generate_caption(It_warp, max_words=2, style=style)
            else:
                caption = generate_keywords(style, max_words=4)

            It_warp = self.pipeline2(caption,
                                     width=warp_img.width, height=warp_img.height,
                                     negative_prompt="worst quality. blurry. indistinct facial features. motion blur. faded colors. abstract. unclear background. washed out. distorted.",
                                     num_inference_steps=NIS, image=warp_img, strength=0.4).images[0]
            It_warp = It_warp.resize(self.config.test_size)
            It_warp = self.trans(It_warp.convert('RGB')).to(self.device).unsqueeze(0)
        # breakpoint()

        return It_warp, F12, F21, F12in, F21in
