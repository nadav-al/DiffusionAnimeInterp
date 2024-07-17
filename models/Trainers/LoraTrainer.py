import os
import shutil
import argparse
try:
    from utils.lora_utils import generate_metadata, preprocess_single_folder, preprocess_multiple_folders, generate_folder, Methods
except:
    from utils import generate_metadata, preprocess_single_folder, preprocess_multiple_folders, generate_folder, Methods

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
        default=6
    )

    args, _ = parser.parse_known_args()

    return args

class LoRATrainer:
    def __init__(self):
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
            raise NameError('Unrecognized base model. Choose between ["sdxl", "sd15", "animagine"]')

        if self.args.multi_gpu:
            self.multi_gpu = "--multi_gpu"
        else:
            self.multi_gpu = ""

        self.methods = [Methods.RANDOM_SIZE, Methods.JITTER_RANDOM]

    def train_single_folder(self, path, seed=0, apply_preprocess=True, max_crops=200, store_preprocess=True, unique_folder=False):
        folder = os.path.split(path)[-1]
        if apply_preprocess:
            path = preprocess_single_folder(path, max_crops=max_crops, methods=self.methods, unique_folder=unique_folder)
        else:
            store_preprocess = True  # We don't want to erase the original data directory

        generate_metadata(path)

        output_path = generate_folder(folder, extension=self.out_path_ext, unique_folder=unique_folder)
        # output_path = os.path.join("checkpoints/outputs/LoRAs/07-09/test13", folder)
        os.system(
            f"accelerate launch {self.multi_gpu} models/Trainers/{self.script_name}.py \
                      --pretrained_model_name_or_path={self.diff_path} \
                      --train_data_dir={path} \
                      --rank={self.args.rank} \
                      --mixed_precision='fp16' \
                      --dataloader_num_workers=8 \
                      --train_batch_size={self.args.train_bs} \
                      {self.train_text_enc} \
                      --learning_rate=1e-04 \
                      --lr_scheduler='cosine' \
                      --snr_gamma=5 \
                      --lr_warmup_steps=0 \
                      --output_dir={output_path} \
                      --num_train_epochs=10 \
                      --checkpointing_steps=50 \
                      --resume_from_checkpoint='latest' \
                      --seed={seed} \
                      --scale_lr"
            )

        if not store_preprocess:
            shutil.rmtree(path)

        return output_path

    def train_multiple_folders(self, path, seed=0, apply_preprocess=True, max_crops=200, store_preprocess=True, unique_folder=False):
        folder = os.path.split(path)[-1]
        if apply_preprocess:
            path = preprocess_multiple_folders(path, max_crops=max_crops, methods=self.methods, unique_folder=unique_folder)
            generate_metadata(path, multi_folder=True)
        else:
            store_preprocess = True
        output_path = generate_folder(folder, extension=self.out_path_ext, unique_folder=unique_folder)
        os.system(
            f"accelerate launch {self.multi_gpu} models/Trainers/{self.script_name}.py \
                      --pretrained_model_name_or_path={self.diff_path} \
                      --train_data_dir={path} \
                      --rank={self.args.rank} \
                      --mixed_precision='fp16' \
                      --dataloader_num_workers=8 \
                      --train_batch_size={self.args.train_bs} \
                      {self.train_text_enc} \
                      --learning_rate=1e-04 \
                      --lr_scheduler='cosine' \
                      --snr_gamma=5 \
                      --lr_warmup_steps=0 \
                      --output_dir={output_path} \
                      --num_train_epochs=10 \
                      --checkpointing_steps=50 \
                      --resume_from_checkpoint='latest' \
                      --seed={seed} \
                      --scale_lr"
        )

        if not store_preprocess:
            shutil.rmtree(path)

        return output_path





