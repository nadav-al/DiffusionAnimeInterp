import os
import shutil
import argparse
from utils.lora_utils import generate_metadata, preprocess, generate_folder, Methods

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
        "--data_path",
        type=str,
        default=None,
        required=True,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        )
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

    args, _ = parser.parse_known_args()

    if args.data_path is None:
        raise ValueError("Need a path for the trainin dataset")

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
            raise ValueError('Unrecognized base model. Choose between ["sdxl", "sd15", "animagine"]')

        if self.args.multi_gpu:
            self.multi_gpu = "--multi_gpu"
        else:
            self.multi_gpu = ""

        self.methods = [Methods.RANDOM_SIZE, Methods.JITTER_RANDOM]

    def train(self, path, apply_preprocess=True, store_preprocess=True, unique_folder=False):
        folder = os.path.split(path)[-1]
        if apply_preprocess:
            path = preprocess(path, size=(300, 300), target_size=512, methods=self.methods, unique_folder=unique_folder)
            generate_metadata(path)
        else:
            store_preprocess = True  # We don't want to erase the original data directory

        output_path = generate_folder(folder, unique_folder=unique_folder)
        output_path = os.path.join("checkpoints/outputs/LoRAs/07-09/test13", folder)
        print(path)
        os.system(
            f"accelerate launch {self.multi_gpu} models/Trainers/{self.script_name}.py \
                      --pretrained_model_name_or_path={self.diff_path} \
                      --train_data_dir={path} \
                      --rank=6 \
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
                      --scale_lr")

        if not store_preprocess:
            shutil.rmtree(output_path)

        return output_path





