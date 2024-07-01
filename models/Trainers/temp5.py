import os
import shutil
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
        default=4
    )
    parser.add_argument(
        "--store_folder",
        action="store_true",
    )
    args = parser.parse_args()

    if args.data_path is None:
        raise ValueError("Need a path for the trainin dataset")

    return args



if __name__ == "__main__":
    args = parse_args()

    if args.use_xl:
        diff_path = "checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0"
        # diff_path = "cagliostrolab/animagine-xl-3.1"
        out_path_ext = "sdxl"
        script_name = "train_text_to_image_lora_sdxl"
    else:
        diff_path = "checkpoints/diffusers/runwayml/stable-diffusion-v1-5"
        out_path_ext = "sd1.5"
        script_name = "train_text_to_image_lora"

    fol = ""
    methods = [Methods.RANDOM_SIZE, Methods.JITTER_RANDOM]
    for idx, path in enumerate(os.listdir(args.data_path)):
        # if idx > 20:
        #     break
        dir = os.path.join(args.data_path, path)
        if idx==0:
            unique=True
        else:
            unique=False

        # processed_data_path = preprocess(dir, size=(225,225), methods=methods, unique_folder=unique)
        processed_data_path = preprocess(dir, size=(300,300), methods=methods, unique_folder=unique, save_folder=args.store_folder)
        print("processed_data_path: ", processed_data_path)
        generate_metadata(processed_data_path)
        output_path = generate_folder(path,unique_folder=unique)
        print("output_path: ", output_path)


        os.system(
            f"accelerate launch --multi_gpu models/Trainers/{script_name}.py \
              --pretrained_model_name_or_path={diff_path} \
              --train_data_dir={dir} \
              --rank=6 \
              --mixed_precision='fp16' \
              --dataloader_num_workers=8 \
              --resolution=512 \
              --train_batch_size={args.train_bs} \
              --train_text_encoder \
              --learning_rate=1e-04 \
              --lr_scheduler='cosine' \
              --lr_warmup_steps=40 \
              --output_dir={output_path} \
              --num_train_epochs=600 \
              --checkpointing_steps=100 \
              --seed=100398 \
              --scale_lr")
        fol = os.path.split(processed_data_path)[0]

    if not args.store_folder:
        shutil.rmtree(fol)
