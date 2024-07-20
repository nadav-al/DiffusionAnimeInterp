import torch
import torch.nn as nn
from .DiffimeInterp import DiffimeInterp
# from .Trainers.AniTrainer import LoraTrainerSimpler
# from .Trainers.CustomTrainer import LoraT
# from .Trainers.temp3 import trainer
from .Trainers.LoraTrainer import LoRATrainer
import os
import json


class LoraInterp(nn.Module):
    """The quadratic model"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.trainer = LoRATrainer()
        self.counter = -1

    def train_lora_multi_scene(self, folder, apply_preprocess=True, store_preprocess=True):
        path = folder
        if apply_preprocess:
            if not os.path.exists(folder):
                path = os.path.join(self.config.testset_root, folder)
        else:
            store_preprocess = True  # we don't want to erase the original data folder
        self.counter += 1
        return self.trainer.train_multi_scene(path, apply_preprocess, store_preprocess, unique_folder=(self.counter == 0))

    def train_lora_single_scene(self, folder, apply_preprocess=True, store_preprocess=True):
        path = folder
        if apply_preprocess:
            if not os.path.exists(folder):
                path = os.path.join(self.config.testset_root, folder)
        else:
            store_preprocess = True  # we don't want to erase the original data folder
        self.counter += 1
        return self.trainer.train_single_scene(path, apply_preprocess, store_preprocess, unique_folder=(self.counter == 0))



    def forward(self, I1, I2, F12i, F21i, t, folder=None, weights_path=None):
        self.base_model.pipeline.load_lora_weights(weights_path, weight_name="pytorch_lora_weights.safetensors")
        outputs = self.base_model.forward(I1, I2, F12i, F21i, t, folder=folder)
        self.base_model.pipeline.unload_lora_weights()
        return outputs










