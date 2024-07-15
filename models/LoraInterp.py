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

    def train(self, folder, store_preprocess=True):


    def forward(self, I1, I2, F12i, F21i, t, folder=None, store_preprocess=True, weights_path=None):
        if weights_path is None or not os.path.exists(weights_path):
            path = os.path.join("TempDatasets/07-09/test1", folder[0][0])
            if os.path.exists(path):
                apply_prep = False
            else:
                path = os.path.join(self.config.testset_root, folder[0][0])
                apply_prep = True
            weights_path = self.trainer.train(path, apply_preprocess=apply_prep, store_preprocess=store_preprocess, unique_folder=(self.counter == 0))
            self.counter += 1
        # self.pipeline.load_lora_weights(weights_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="lora")
        # # self.pipeline.set_adapters("lora")
        # outputs = DiffimeInterp.forward(self, I1, I2, F12i, F21i, t, folder=folder)
        # self.pipeline.unload_lora_weights()
        # return outputs
        return None










