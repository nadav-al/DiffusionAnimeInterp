import os
import shutil
import argparse
from utils import generate_metadata, preprocess, generate_folder, Methods

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
        action="store_true"
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

    if args.base_model == "sdxl":
        diff_path = "checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0"
        out_path_ext = "sdxl"
        script_name = "train_text_to_image_lora_sdxl"
        train_text_enc = "--train_text_encoder"
    elif args.base_model == "animagine":
        diff_path = "checkpoints/diffusers/cagliostrolab/animagine-xl-3.1"
        out_path_ext = "animagine"
        script_name = "train_text_to_image_lora_sdxl"
        train_text_enc = "--train_text_encoder"
    elif args.base_model == "sd15":
        diff_path = "checkpoints/diffusers/runwayml/stable-diffusion-v1-5"
        out_path_ext = "sd1.5"
        script_name = "train_text_to_image_lora"
        train_text_enc = ""

    if args.multi_gpu:
        multi_gpu = "--multi_gpu"
    else:
        multi_gpu = ""
    
    fol = ""
    methods = [Methods.RANDOM_SIZE, Methods.JITTER_RANDOM]
    for idx, path in enumerate(os.listdir(args.data_path)):
        # if idx > 20:
        #     break
        if idx == 0:
            continue
        dir = os.path.join(args.data_path, path)
        if idx == 1:
            unique=True
        else:
            unique=False

        # processed_data_path = preprocess(dir, size=(225,225), methods=methods, unique_folder=unique)
        processed_data_path = preprocess(dir, size=(300, 300), methods=methods, unique_folder=unique, save_folder=args.store_folder)
        print("processed_data_path: ", processed_data_path)
        generate_metadata(processed_data_path)
        output_path = generate_folder(path, unique_folder=unique)
        print("output_path: ", output_path)

        os.system(
            f"accelerate launch {multi_gpu} models/Trainers/{script_name}.py \
                              --pretrained_model_name_or_path={diff_path} \
                              --train_data_dir={path} \
                              --rank=6 \
                              --mixed_precision='fp16' \
                              --dataloader_num_workers=8 \
                              --train_batch_size={args.train_bs} \
                              {train_text_enc} \
                              --learning_rate=1e-04 \
                              --lr_scheduler='cosine' \
                              --snr_gamma=5 \
                              --lr_warmup_steps=0 \
                              --output_dir={output_path} \
                              --num_train_epochs=10 \
                              --checkpointing_steps=50 \
                              --resume_from_checkpoint='latest' \
                              --scale_lr")
        fol = os.path.split(processed_data_path)[0]

    if not args.store_folder:
        shutil.rmtree(fol)
