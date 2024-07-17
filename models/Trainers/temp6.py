from LoraTrainer import LoRATrainer
import argparse
from utils import preprocess_multiple_folders
def parse_args():
    parser = argparse.ArgumentParser()
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
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    trainer = LoRATrainer()

    weights_path = trainer.train_multiple_folders(args.data_path, apply_preprocess=True, max_crops=500, unique_folder=True)
    # path = preprocess_multiple_folders(args.data_path, save_folder=True)

    print("success, ", weights_path)