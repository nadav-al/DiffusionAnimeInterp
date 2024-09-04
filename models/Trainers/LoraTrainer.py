import os
import shutil
import argparse
from utils.image_processing import preprocess_single_scene, preprocess_multiple_scenes
from utils.files_and_folders import generate_folder, repeat_data
from utils.captionning import generate_metadata
# from utils2.config import Config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="sdxl",
        help='The base model to use. Choose between ["sdxl", "sd15", "animagine"]',
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoints/outputs/LoRAs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_bs",
        type=int,
        default=1,
        help="The size of the batch size for training",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=6,
        help="The rank of the LoRA",
    )
    parser.add_argument(
        "--data_repeats",
        type=int,
        default=1,
    )

    args, _ = parser.parse_known_args()

    return args

class LoRATrainer:
    def __init__(self, config):
        self.args = parse_args()
        if self.args.base_model == "sdxl":
            self.diff_path = "checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0"
            self.out_path_ext = "sdxl"
            self.script_name = "train_text_to_image_lora_sdxl"
            self.train_text_enc = "--train_text_encoder"
        elif self.args.base_model == "animagine":
            self.diff_path = "checkpoints/diffusers/cagliostrolab/animagine-xl-3.1"
            self.out_path_ext = "animagine"
            self.script_name = "train_text_to_image_lora_sdxl"
            self.train_text_enc = "--train_text_encoder"
        elif self.args.base_model == "sd15":
            self.diff_path = "checkpoints/diffusers/runwayml/stable-diffusion-v1-5"
            self.out_path_ext = "sd1.5"
            self.script_name = "train_text_to_image_lora"
            self.train_text_enc = ""
        else:
            raise ValueError('Unrecognized base model. Choose between ["sdxl", "sd15", "animagine"]')

        if self.args.multi_gpu:
            self.multi_gpu = "--multi_gpu"
        else:
            self.multi_gpu = ""

        self.config = config
        # self.methods = [Methods.RANDOM_SIZE, Methods.JITTER_RANDOM]


    def train_multi_scene(self, data_path, folder_base, test_details, folder_name=None, weights_path=None, apply_preprocess=True, store_preprocess=True, unique_folder=False, only_2_frames=True, with_caption=True):
        flag = False
        if weights_path is not None:
            folders = weights_path.split('/')
            if folders[-1].startswith("test"):
                data_path = generate_folder(None, folder_base=folder_base, root_path="TempDatasets", test_details=test_details, unique_folder=False)
            else:
                data_path = generate_folder(folders[-1], folder_base=folder_base, root_path="TempDatasets", test_details=folders[-2], unique_folder=False)
        else:
            if apply_preprocess:
                flag=True
                data_path = preprocess_multiple_scenes(data_path, folder_base=folder_base, test_details=test_details, unique_folder=unique_folder, only_2_frames=only_2_frames)
            if self.args.data_repeats > 1:
                if flag:
                    data_path = repeat_data(data_path, self.args.data_repeats, folder_name=folder_name, folder_base=folder_base, test_details=test_details, output_folder=data_path, unique_folder=unique_folder)
                else:
                    data_path = repeat_data(data_path, self.args.data_repeats, folder_name=folder_name, folder_base=folder_base,
                                            test_details=test_details, unique_folder=unique_folder)
                    flag=True

        if flag:
            generate_metadata(data_path, with_caption=with_caption)
        else:
            store_preprocess = True  # We don't want to erase the original data directory
        if weights_path is None:
            weights_path = generate_folder(folder_name=folder_name, folder_base=folder_base, test_details=test_details, unique_folder=unique_folder)

        os.system(
            f"accelerate launch {self.multi_gpu} \
                    models/Trainers/{self.script_name}.py \
                              --pretrained_model_name_or_path={self.diff_path} \
                              --train_data_dir={data_path} \
                              --gradient_accumulation_steps=1 \
                              --rank={self.args.rank} \
                              --local_rank=-1 \
                              --mixed_precision='fp16' \
                              --dataloader_num_workers=8 \
                              --train_batch_size={self.args.train_bs} \
                              {self.train_text_enc} \
                              --learning_rate=1e-04 \
                              --lr_scheduler='cosine_with_restarts' \
                              --snr_gamma=5 \
                              --lr_warmup_steps=0 \
                              --output_dir={weights_path} \
                              --max_train_steps=4000 \
                              --checkpointing_steps=50 \
                              --resume_from_checkpoint='latest'")

        # Added gradient_accumulation_steps
        # Changed lr_scheduler from 'cosine' to 'cosine_with_restarts'
        # Changed dataloader_num_workers from 8 to 0
        # Added local_rank
        # Removed lr_scale
        # Removed num_train_epochs=100
        # Added max_train_steps
        # Changed checkpointing_steps from 50 to 150


        if not store_preprocess:
            shutil.rmtree(data_path)

        return weights_path


    def train_single_scene(self, data_path, folder_name=None, folder_base=None, test_details=None, weights_path=None, apply_preprocess=True, store_preprocess=True, unique_folder=False, only_2_frames=True, with_caption=True):
        flag = False
        data_repeats = self.args.data_repeats
        if weights_path is not None:
            folders = weights_path.split('/')
            if folders[-1].startswith("test"):
                path = generate_folder(folder_name, folder_base=folder_base, root_path="TempDatasets", test_details=test_details, unique_folder=False, create=False)
                if folder_name is not None:
                    weights_path = os.path.join(weights_path, folder_name)
            else:
                path = generate_folder(folders[-1], folder_base=folder_base, root_path="TempDatasets", test_details=folders[-2], unique_folder=False, create=False)

            if os.path.exists(path) and len(os.listdir(path)) > 0:
                data_path = path
                apply_preprocess = False
                data_repeats = 0

        else:
            weights_path = generate_folder(folder_name=folder_name, folder_base=folder_base,
                                           test_details=test_details, unique_folder=unique_folder)
        if apply_preprocess:
            flag = True
            data_path = preprocess_single_scene(data_path, folder_name=folder_name, folder_base=folder_base, test_details=test_details,
                                                unique_folder=unique_folder, only_2_frames=only_2_frames)
        if data_repeats > 1:
            flag = True
            data_path = repeat_data(data_path, data_repeats, folder_name=folder_name, folder_base=folder_base, test_details=test_details,
                                    unique_folder=unique_folder)
        if flag:
            generate_metadata(data_path, with_caption=with_caption)
        else:
            store_preprocess = True  # We don't want to erase the original data directory


        print("data_path for lora is: ", data_path)
        os.system(
            f"accelerate launch {self.multi_gpu} \
                models/Trainers/{self.script_name}.py \
                      --pretrained_model_name_or_path={self.diff_path} \
                      --train_data_dir={data_path} \
                      --gradient_accumulation_steps=1 \
                      --rank={self.args.rank} \
                      --local_rank=-1 \
                      --mixed_precision='fp16' \
                      --dataloader_num_workers=8 \
                      --train_batch_size={self.args.train_bs} \
                      {self.train_text_enc} \
                      --learning_rate=1e-04 \
                      --lr_scheduler='cosine_with_restarts' \
                      --snr_gamma=5 \
                      --lr_warmup_steps=0 \
                      --output_dir={weights_path} \
                      --max_train_steps=4000 \
                      --checkpointing_steps=50 \
                      --resume_from_checkpoint='latest'")


        if not store_preprocess:
            shutil.rmtree(data_path)

        return weights_path




