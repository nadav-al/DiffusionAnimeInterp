import torch
from .DiffimeInterp import DiffimeInterp
# from .Trainers.AniTrainer import LoraTrainerSimpler
# from .Trainers.CustomTrainer import LoraT
# from .Trainers.temp3 import trainer
from .Trainers.LoraTrainer import LoRATrainer
import os
import json


class LoraInterp(DiffimeInterp):
    """The quadratic model"""
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, args=None):
        super().__init__(path, config, init_diff=True, args=args)
        # self.trainer = LoraTrainerSimpler(config, args)
        self.trainer = LoRATrainer()
        self.counter = 0

    def forward(self, I1, I2, F12i, F21i, t, folder=None, store_preprocess=True, weights_path=None):
        if weights_path is None:
            path = os.path.join("TempDatasets/07-09/test1", folder[0][0])

            if os.path.exists(path):
                apply_prep = False
            else:
                path = os.path.join(self.config.testset_root, folder[0][0])
                apply_prep = True
            weights_path = self.trainer.train(path, apply_preprocess=apply_prep, store_preprocess=store_preprocess, unique_folder=(self.counter == 0))
            self.counter += 1
        else:
            weights_path = os.path.join(weights_path, folder[0][0])

        self.pipeline.load_lora_weights(weights_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="lora")
        # self.pipeline.set_adapters("lora")
        outputs = super().forward(I1, I2, F12i, F21i, t, folder)
        self.pipeline.unload_lora_weights()

        return outputs


    # def forward(self, I1, I2, F12i, F21i, t, folder=None):
    #     # extract features
    #     I1o, features1, I2o, features2 = self.extract_features_2_frames(I1, I2)
    #     feat11, feat12, feat13 = features1
    #     feat21, feat22, feat23 = features2
    #
    #     # calculate motion
    #     F12, F12in, F1ts = self.motion_calculation(I1o, I2o, F12i, [feat11, feat12, feat13], t, 0)
    #     F21, F21in, F2ts = self.motion_calculation(I2o, I1o, F21i, [feat21, feat22, feat23], t, 1)
    #
    #     # warping
    #     I1t, feat1t, norm1, norm1t = self.warping(F1ts, I1, features1)
    #     I2t, feat2t, norm2, norm2t = self.warping(F2ts, I2, features2)
    #
    #     # normalize
    #     # Note: normalize in this way benefit training than the original "linear"
    #     self.normalize(I1t, feat1t, norm1, norm1t)
    #     self.normalize(I2t, feat2t, norm2, norm2t)
    #
    #
    #     # diffusion
    #     # combined_images = torch.cat((I1, I2), dim=0)
    #     # # self.trainer.train_from_tensors(combined_images, folder)
    #     # self.trainer.train(combined_images, folder)
    #
    #     It_warp = self.synnet(torch.cat([I1t, I2t], dim=1), torch.cat([feat1t[0], feat2t[0]], dim=1),
    #                           torch.cat([feat1t[1], feat2t[1]], dim=1),
    #                           torch.cat([feat1t[2], feat2t[2]], dim=1))
    #
    #     # Improve quality of It_warp using diffusion model
    #     # TODO: something like this:
    #     if self.LoRA_weights_path is None:
    #         directory = os.path.join(self.config.testset_root, folder[0][0])
    #         weights_path = self.trainer.train(directory, unique_folder=True)
    #     else:
    #         weights_path = self.LoRA_weights_path
    #     self.pipeline.load_lora_weights(weights_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="lora")
    #     self.pipeline.set_adapters("lora")
    #
    #     It_warp = self.revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)
    #     metadata_file = os.path.join(self.config.testset_root, folder[0][0], "metadata.jsonl")
    #     with open(metadata_file, 'r') as f:
    #         line = json.loads(f.readline())
    #         prompt = line["text"]
    #     It_warp = self.pipeline(prompt,
    #                             num_inferece_steps=25, image=It_warp).images[0]
    #     It_warp = It_warp.resize(self.config.test_size)
    #     It_warp = self.trans(It_warp.convert('RGB')).to(self.device).unsqueeze(0)
    #
    #
    #     return It_warp, F12, F21, F12in, F21in







