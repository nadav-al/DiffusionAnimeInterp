import os
import argparse
from utils import generate_metadata, preprocess, generate_folder, Methods

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_xl",
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
        default=8,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()

    if args.data_path is None:
        raise ValueError("Need a path for the trainin dataset")

    return args

class LoRATrainer:
    def __init__(self):
        self.args = parse_args()
        if self.args.use_xl:
            self.diff_path = "checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0"
            self.out_path_ext = "sdxl"
            self.script_name = "train_text_to_image_lora_sdxl"
        if self.args.animagine:
            self.diff_path = "checkpoints/diffusers/cagliostrolab/animagine-xl-3.1"
            self.out_path_ext = "animagine"
            self.script_name = "train_text_to_image_lora_sdxl"
        else:
            self.diff_path = "checkpoints/diffusers/runwayml/stable-diffusion-v1-5"
            self.out_path_ext = "sd1.5"
            self.script_name = "train_text_to_image_lora"

        self.methods = [Methods.RANDOM_SIZE, Methods.JITTER_RANDOM]

    def train(self, path, apply_preprocess=True, store_preprocess=True, unique_folder=False):
        directory = os.path.join(self.args.data_path, path)
        if apply_preprocess:
            directory = preprocess(directory, size=(256, 256), target_size=512, methods=self.methods, unique_folder=unique_folder)
        else:
            store_preprocess = True  # We don't want to erase the original data directory




