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
        self.trainer = LoRATrainer()
        self.base_model = base_model

    def train_lora_from_single_folder(self, folder, seed=0, max_crops=200, store_preprocess=True, unique_folder=False):
        path = os.path.join("TempDatasets/07-17/test1", folder)
        if os.path.exists(path):
            apply_prep = False
        else:
            path = os.path.join(self.base_model.config.testset_root, folder)
            apply_prep = True
        print(path)
        return self.trainer.train_single_folder(path, seed=seed, max_crops=max_crops, apply_preprocess=apply_prep,
                                                store_preprocess=store_preprocess, unique_folder=unique_folder)

    def train_lora_from_multi_folder(self, root=None, seed=0, max_crops=200, store_preprocess=True, unique_folder=False):
        if not root:
            path = self.base_model.config.testset_root
            apply_prep = True
        else:
            path = os.path.join("TempDatasets/07-17/test1", os.path.split(root)[-1])
            if os.path.exists(path):
                apply_prep = False
            else:
                path = root
                apply_prep = True
        return self.trainer.train_multiple_folders(path, seed=seed, max_crops=max_crops, apply_preprocess=apply_prep,
                                                   store_preprocess=store_preprocess, unique_folder=unique_folder)

    def forward(self, I1, I2, F12i, F21i, t, folder=None, weights_path=None):
        if weights_path is None:
            raise OSError("Path to LoRA weights must be given")
        self.base_model.pipeline.load_lora_weights(weights_path, weight_name="pytorch_lora_weights.safetensors")
        outputs = self.base_model.forward(I1, I2, F12i, F21i, t, folder=folder)
        self.base_model.pipeline.unload_lora_weights()
        return outputs










