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


    def train_multi_scene(self, path, apply_preprocess=True, store_preprocess=True, unique_folder=False, only_2_frames=True):
        folder = os.path.split(path)[-1]
        if apply_preprocess:
            path = preprocess_multiple_scenes(path, unique_folder=unique_folder, only_2_frames=only_2_frames)
            generate_metadata(path)
        elif self.args.data_repeats > 1:
            path = repeat_data(path, self.args.data_repeats, unique_folder=unique_folder)
            generate_metadata(path)
        else:
            store_preprocess = True  # We don't want to erase the original data directory

        output_path = generate_folder(folder, extension=f"rank{self.args.rank}", unique_folder=unique_folder)
        width, height = self.config.test_size
        os.system(
            f"accelerate launch {self.multi_gpu} models/Trainers/{self.script_name}.py \
                              --pretrained_model_name_or_path={self.diff_path} \
                              --train_data_dir={path} \
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
                              --output_dir={output_path} \
                              --max_train_steps=4000 \
                              --checkpointing_steps=150 \
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
            shutil.rmtree(path)

        return output_path


    def train_single_scene(self, path, apply_preprocess=True, store_preprocess=True, unique_folder=False, only_2_frames=True):
        folder = os.path.split(path)[-1]
        if folder in [ 'lora', 'frames' ]:
            folder = os.path.split(os.path.split(path)[0])[-1]
        if apply_preprocess:
            path = preprocess_single_scene(path, unique_folder=unique_folder, only_2_frames=only_2_frames)
            generate_metadata(path)
        elif self.args.data_repeats > 1:
            path = repeat_data(path, self.args.data_repeats, unique_folder=unique_folder)
            generate_metadata(path)
        else:
            store_preprocess = True  # We don't want to erase the original data directory

        output_path = generate_folder(folder, extension=f"rank{self.args.rank}", unique_folder=unique_folder)
        width, height = self.config.test_size
        print("data_path for lora is: ", path)
        os.system(
            f"accelerate launch {self.multi_gpu} models/Trainers/{self.script_name}.py \
                      --pretrained_model_name_or_path={self.diff_path} \
                      --train_data_dir={path} \
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
                      --output_dir={output_path} \
                      --max_train_steps=4000 \
                      --checkpointing_steps=150 \
                      --resume_from_checkpoint='latest'")

        if not store_preprocess:
            shutil.rmtree(path)

        return output_path


if __name__ == '__main__':
    print("Started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=True)
    parser.add_argument("--lora_data_path",
                        type=str,
                        required=True)
    parser.add_argument('--multiscene',
                        action="store_true",)
    parser.add_argument(
        "-p",
        action="store_true",
        help="Apply preprocess on the data"
    )

    args, _ = parser.parse_known_args()
    config = Config.from_file(args.config)

    trainer = LoRATrainer(config)
    print("Trainer constructed")
    if args.multiscene:
        output_path = trainer.train_multi_scene(args.lora_data_path, args.p, unique_folder=True)
        print(output_path)
    else:
        i = 0
        for folder_name in os.listdir(args.lora_data_path):
            folder = os.path.join(args.lora_data_path, folder_name)
            if not os.isdir(folder):
                continue
            if i == 0:
                output_path = trainer.train_single_scene(folder, args.p, unique_folder=True)
            else:
                output_path = trainer.train_single_scene(folder, args.p)

            print(output_path)
            i += 1




