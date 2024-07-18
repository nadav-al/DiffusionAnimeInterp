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
from utils.image_processing import generate_caption

from transformers import CLIPVisionModelWithProjection
from diffusers import ControlNetModel, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers import StableDiffusionImg2ImgPipeline

from PIL import Image

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


class CannyDiffimeInterp(DiffimeInterp):
    """The quadratic model"""
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, canny_th_low=50, canny_th_high=200, args=None):
        super(CannyDiffimeInterp, self).__init__(path, config, False, args)
        self.cth_low = canny_th_low
        self.cth_heigh = canny_th_high

        print("loading controlnet")
        # controlnet = None
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        self.load_diffuser("text", controlnet)
        print("controlnet loaded")

        # image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        #     "h94/IP-Adapter",
        #     subfolder="models/image_encoder",
        #     torch_dtype=torch.float16
        # )
        #
        # self.pipeline = AutoPipelineForText2Image.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0",
        #     image_encoder=image_encoder,
        #     torch_dtype=torch.float16
        # ).to("cuda")
        #
        # self.pipeline.load_ip_adapter(config.ip_adapter_id, subfolder="sdxl_models",
        #                               weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
        # self.pipeline.image_encoder.to("cuda")
        # self.load_diffuser(type="text", controlnet=controlnet)



        print("loading adapter")
        # self.pipeline.load_ip_adapter(config.ip_adapter_id, subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')
        # self.pipeline.set_ip_adapter_scale(1.0)
        # self.pipeline.image_encoder.to("cuda")
        print("adapter loaded")


    def forward(self, I1, I2, F12i, F21i, t, folder=None):
        store_latents_path = os.path.join(self.config.store_path, folder[0][0])
        if not os.path.exists(store_latents_path):
            os.mkdir(store_latents_path)
        store_latents_path = os.path.join(store_latents_path, "latents")
        if not os.path.exists(store_latents_path):
            os.mkdir(store_latents_path)


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

        print(w_I1c.shape)
        print(w_I2c.shape)
        w_I1c_img = self.revtrans(w_I1c.cpu()[0])
        w_I1c_img.save(store_latents_path + "/I1_canny.png")
        w_I2c_img = self.revtrans(w_I2c.cpu()[0])
        w_I2c_img.save(store_latents_path + "/I2_canny.png")

        print("canny created")

        # self.pipeline.load_lora_weights("checkpoints/outputs/LoRAs/07-09/test13/Japan_AoT_S4E16_shot1", weight_name="pytorch_lora_weights.safetensors", adapter_name="lora")
        # resizer = TF.Resize((512,512))
        # print(w_I2c_img)
        # I1_im = self.revNormalize(I1.cpu()[0]).unsqueeze(0).to("cuda")
        folder_path = self.config.testset_root + folder[0][0]
        images_path = os.listdir(folder_path)
        im1 = Image.open(os.path.join(folder_path,images_path[0]))#.resize((512,512))
        im2 = Image.open(os.path.join(folder_path,images_path[2]))#.resize((512,512))
        # I1_im = self.pipeline.prepare_ip_adapter_image_embeds(
        #     ip_adapter_image=im1,
        #     ip_adapter_image_embeds=None,
        #     device="cuda",
        #     num_images_per_prompt=1,
        #     do_classifier_free_guidance=True,
        # )
        # I2_im = self.revNormalize(I2.cpu()[0]).unsqueeze(0).to("cuda")
        # I2_im = self.pipeline.prepare_ip_adapter_image_embeds(
        #     ip_adapter_image=im2,
        #     ip_adapter_image_embeds=None,
        #     device="cuda",
        #     num_images_per_prompt=1,
        #     do_classifier_free_guidance=True,
        # )
        # diffuser
        try:
            lora_path = os.path.join("checkpoints/outputs/LoRAs/07-09/test13/", folder[0][0], "checkpoint-500", "pytorch_lora_weights.safetensors")
            self.pipeline.load_lora_weights(lora_path)
        except Exception as e:
            print(f"Error processing {folder[0][0]}: {str(e)}")

        # print("w_I1c_img: ", w_I1c_img.device)
        # print("I1_im: ", I1_im.device)
        # d_I1c = self.pipeline("", image=w_I1c_img, ip_adapter_image_embeds=I1_im).images[0]
        # d_I1c = self.pipeline("", image=w_I1c_img, ip_adapter_image_embeds=I1_im).images[0]
        # d_I1c = self.pipeline("a girl sits on wooden stairs in a basement with barrels behind her. she looks sad and ashamed. Anime", image=w_I2c_img, ip_adapter_image=im1).images[0].resize(self.config.test_size)
        # d_I2c = self.pipeline("a girl sits on wooden stairs in a basement with barrels behind her. she looks sad and ashamed. Anime ", image=w_I2c_img, ip_adapter_image=im2).images[0].resize(self.config.test_size)
        # d_I1c = self.pipeline("best quality, high quality, in the style of AniInt", image=w_I1c_img, ip_adapter_image=im1).images[0].resize(self.config.test_size)
        # d_I2c = self.pipeline("best quality, high quality, in the style of AniInt", image=w_I2c_img, ip_adapter_image=im2).images[0].resize(self.config.test_size)
        caption1 = generate_caption(im1, max_words=3, style="Anime")
        d_I1c = self.pipeline(caption1, image=w_I1c_img).images[0].resize(self.config.test_size)
        d_I2c = self.pipeline(caption1, image=w_I2c_img).images[0].resize(self.config.test_size)
        self.pipeline.unload_lora_weights()
        print("diffused")

        # for exploration and understanding the model, saves the intermediate results
        # self.revtrans(w_I1c.cpu()[0]).save(f'{self.store_path}/canny_I1_{self.counter}.png')
        # self.revtrans(w_I2c.cpu()[0]).save(f'{self.store_path}/canny_I2_{self.counter}.png')
        print(store_latents_path)
        d_I1c.save(f'{store_latents_path}/lat_frame1.png')
        d_I2c.save(f'{store_latents_path}/lat_frame3.png')

        d_I1c = self.trans(d_I1c).unsqueeze(0).to("cuda")
        d_I2c = self.trans(d_I2c).unsqueeze(0).to("cuda")


        self.normalize(d_I1c, feat1t, norm1, norm1t)
        self.normalize(d_I2c, feat2t, norm2, norm2t)
        
        # synthesis
        It_warp = self.synnet(torch.cat([d_I1c, d_I2c], dim=1), torch.cat([feat1t[0], feat2t[0]], dim=1),
                              torch.cat([feat1t[1], feat2t[1]], dim=1), torch.cat([feat1t[2], feat2t[2]], dim=1))
        #
        warp_im = TF.ToPILImage()(self.revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))
        warp_im.save(f"{store_latents_path}/frame2.png")

        return It_warp, F12, F21, F12in, F21in
