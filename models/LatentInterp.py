import torch
import numpy as np
from models.DiffimeInterp import DiffimeInterp

class LatentInterp(DiffimeInterp):
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, args=None):
        super(LatentInterp, self).__init__(path=path, config=config, args=args)
        self.load_diffuser()
        self.beta = self.pipeline.scheduler.config.beta_start

    def __slerp(self, v0, v1, t, DOT_THESHOLD=0.9995):
        """helper functions to spherically interpolate two arrays v0, v1"""
        dot = np.sum([v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1))])
        if np.abs(dot) > DOT_THESHOLD:
            v2 = (1-t)*v0 + t*v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = t*theta_0
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0*v0 + s1*v1

        return v2

    def forward(self, I1, I2, F12i, F21i, t):
        # extract features
        I1o, features1, I2o, features2 = self.extract_features_2_frames(I1, I2)
        feat11, feat12, feat13 = features1
        feat21, feat22, feat23 = features2

        # calculate motion
        F12, F12in, F1t, F1td, F1tdd, F1tddd = self.motion_calculation(I1o, I2o, F12i, [feat11, feat12, feat13], t, 0)
        F21, F21in, F2t, F2td, F2tdd, F2tddd = self.motion_calculation(I2o, I1o, F21i, [feat21, feat22, feat23], t, 1)

        img_I1o = self.revtrans(I1o.cpu()[0])
        img_I2o = self.revtrans(I2o.cpu()[0])
        tens_I1o = self.pipeline.image_processor.preprocess(img_I1o)
        tens_I2o = self.pipeline.image_processor.preprocess(img_I2o)

        num_inference_steps = self.pipeline.scheduler.timesteps[:1].repeat(1)
        device = self.pipeline._execution_device
        lat1 = self.pipeline.prepare_latents(tens_I1o, num_inference_steps, 1, 1, None, device, generator=self.generator, add_noise=True)
        lat2 = self.pipeline.prepare_latents(tens_I2o, num_inference_steps, 1, 1, None, device, generator=self.generator, add_noise=True)
        # lat2 = self.pipeline.vae.encode(I2o)

        lat_interp = self.__slerp(lat1.cpu()[0], lat2.cpu()[0], t)
        # lat_interp = lat_interp.permute(1, 2, 0)
        lat_interp = lat_interp.float()
        # print(lat_interp.shape)
        dims = self.config.test_size
        It_warp = self.pipeline('', image=lat_interp)
        It_warp = self.trans(It_warp.convert('RGB')).to(self.device).unsqueeze(0)

        return It_warp, F12, F21, F12in, F21in



