from .DiffimeInterp import DiffimeInterp
from .CannyDiffimeInterp import CannyDiffimeInterp
from .Trainers.LoraTrainer import LoRATrainer
import os


class LoraInterp(DiffimeInterp):
    """The quadratic model"""
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, init_diff=True, args=None):
        super().__init__(path, config, init_diff, args)
        self.trainer = LoRATrainer(config)
        self.counter = -1

    def train_lora_multi_scene(self, folder, folder_base, test_details, weights_path=None, apply_preprocess=True, store_preprocess=True, with_caption=True):
        path = folder
        if apply_preprocess:
            if not os.path.exists(folder):
                path = os.path.join(self.config.testset_root, folder)
        else:
            store_preprocess = True  # we don't want to erase the original data folder
        self.counter += 1
        print(path)
        return self.trainer.train_multi_scene(path, folder_base=folder_base, test_details=test_details, weights_path=weights_path, apply_preprocess=apply_preprocess, store_preprocess=store_preprocess, with_caption=with_caption, unique_folder=(self.counter == 0))

    def train_lora_single_scene(self, folder, folder_base, test_details, folder_name=None, weights_path=None, apply_preprocess=True, store_preprocess=True, with_caption=True):
        path = folder
        if apply_preprocess:
            if not os.path.exists(folder):
                path = os.path.join(self.config.testset_root, folder)
        else:
            store_preprocess = True  # we don't want to erase the original data folder
        self.counter += 1
        print(path)
        return self.trainer.train_single_scene(path, folder_name=folder_name, folder_base=folder_base, test_details=test_details, weights_path=weights_path, apply_preprocess=apply_preprocess, store_preprocess=store_preprocess, with_caption=with_caption, unique_folder=(self.counter == 0))


    def forward(self, I1, I2, F12i, F21i, t, folder=None, weights_path=None):
        test_details = ""
        weights_folders = weights_path.split('/')
        for i in range(len(weights_folders)-1, -1, -1):
            if weights_folders[i].startswith("test"):
                test_details = weights_folders[i]
                break
        self.pipeline.load_lora_weights(weights_path, weight_name="pytorch_lora_weights.safetensors")

        outputs = super().forward(I1, I2, F12i, F21i, t, folder=folder, test_details=test_details)
        self.pipeline.unload_lora_weights()
        return outputs


class LoraCNInterp(CannyDiffimeInterp, LoraInterp):
    """The quadratic model"""
    def __init__(self, path='models/raft_model/models/rfr_sintel_latest.pth-no-zip', config=None, init_diff=True, args=None):
        super().__init__(path, config, init_diff, args)

    def forward(self, I1, I2, F12i, F21i, t, folder=None, weights_path=None):
        test_details = ""
        weights_folders = weights_path.split('/')
        for i in range(len(weights_folders) - 1, -1, -1):
            if weights_folders[i].startswith("test"):
                test_details = weights_folders[i]
                break
        self.pipeline.load_lora_weights(weights_path, weight_name="pytorch_lora_weights.safetensors")
        if self.config.diff_objective == "both":
            self.pipeline2.load_lora_weights(weights_path, weight_name="pytorch_lora_weights.safetensors")
        outputs = super().forward(I1, I2, F12i, F21i, t, folder=folder, test_details=test_details)
        self.pipeline.unload_lora_weights()
        if self.config.diff_objective == "both":
            self.pipeline2.unload_lora_weights()
        return outputs



