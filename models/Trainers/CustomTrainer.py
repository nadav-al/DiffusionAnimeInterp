import argparse
import os
import random

from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, AutoTokenizer, get_scheduler
from torchvision import transforms as TF
from torchvision.transforms.functional import crop
from datasets import load_dataset, Dataset, Features, Image, Value
from datas.LoraDataset import LoraDataset

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--to_save", action="store_true")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=150,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=150,
        )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--image_repeats", type=int, default=10, help="Repeat the images in the training set")


    if input_args is not None:
        args, _ = parser.parse_known_args(input_args)
    else:
        args, _ = parser.parse_known_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



class LoraT:
    def __init__(self, config, args):
        self.config = config
        self.args = parse_args()
        self.unet = UNet2DConditionModel.from_pretrained(self.config.diff_path, subfolder="unet",
                                                         revision=self.args.revision)
        self.unet.requires_grad_(False)
        self.unet.to("cuda")
        self.tokenizer_one = AutoTokenizer.from_pretrained(self.config.diff_path, subfolder="tokenizer",
                                                           revision=self.args.revision, use_fast=False)
        self.tokenizer_two = AutoTokenizer.from_pretrained(self.config.diff_path, subfolder="tokenizer_2",
                                                           revision=self.args.revision, use_fast=False)
        self.preprocess = TF.Compose([
            TF.Resize((self.args.resolution, self.args.resolution)),
            TF.RandomHorizontalFlip(),
            TF.ToTensor(),
            TF.Normalize([0.5], [0.5]),
        ])

    def __prepare_dataloader(self, input):
        ds = LoraDataset(input)
        return ds, DataLoader(ds, shuffle=True, batch_size=self.args.train_batch_size,
                              num_workers=self.args.dataloader_num_workers)

    def create_dataset(self, path):
        data = {
            "image": [],
            "prompt": []
        }
        for i, filename in enumerate(os.listdir(path)):
            if filename.endswith(".png"):
                for i in range(self.args.image_repeats):
                    data["image"].append(os.path.join(path, filename))
                    data["prompt"].append("an image of AnimeInterp")

        features = Features({
            "image": Image(),
            "prompt": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)

        def tokenize_prompt(tokenizer, prompt):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            return text_input_ids

        # We need to tokenize input prompts and transform the images.
        def tokenize_prompts(examples, is_train=True):
            prompts = examples["prompt"]
            tokens_one = tokenize_prompt(self.tokenizer_one, prompts)
            tokens_two = tokenize_prompt(self.tokenizer_two, prompts)
            return tokens_one, tokens_two

        # Preprocessing the datasets.
        train_resize = TF.Resize(self.args.resolution, interpolation=TF.InterpolationMode.BILINEAR)
        train_crop = TF.CenterCrop(self.args.resolution) if self.args.center_crop else TF.RandomCrop(
            self.args.resolution)
        train_flip = TF.RandomHorizontalFlip(p=1.0)
        train_transforms = TF.Compose(
            [
                TF.ToTensor(),
                TF.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples["image"]]
            # image aug
            original_sizes = []
            all_images = []
            crop_top_lefts = []
            for image in images:
                original_sizes.append((image.height, image.width))
                image = train_resize(image)
                if self.args.random_flip and random.random() < 0.5:
                    # flip
                    image = train_flip(image)
                if self.args.center_crop:
                    y1 = max(0, int(round((image.height - self.args.resolution) / 2.0)))
                    x1 = max(0, int(round((image.width - self.args.resolution) / 2.0)))
                    image = train_crop(image)
                else:
                    y1, x1, h, w = train_crop.get_params(image, (self.args.resolution, self.args.resolution))
                    image = crop(image, y1, x1, h, w)
                crop_top_left = (y1, x1)
                crop_top_lefts.append(crop_top_left)
                image = train_transforms(image)
                all_images.append(image)

            # examples["original_sizes"] = original_sizes
            # examples["crop_top_lefts"] = crop_top_lefts
            examples["sample"] = all_images
            tokens_one, tokens_two = tokenize_prompts(examples)
            examples["input_ids_one"] = tokens_one
            examples["input_ids_two"] = tokens_two
            return examples

        ds = ds.with_transform(preprocess_train)

        return ds

    # def tokenize_captions(self, examples):
    #     captions = examples["caption"]
    #     inputs = self.tokenizer(captions, padding="max_length", truncation=True, max_length=77)
    #     examples["caption"] = inputs
    #     return examples

    def train(self, tensors, folder, test_size=0.1):
        folder_path = self.config.testset_root + folder[0][0]
        dataset = self.create_dataset(folder_path)
        train_test_split = dataset.train_test_split(test_size=test_size)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

        def collate_fn(examples):
            pixel_values = torch.stack([example["sample"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            # original_sizes = [example["original_sizes"] for example in examples]
            # crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
            input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
            return {
                "sample": pixel_values,
                # "input_ids_one": input_ids_one,
                # "input_ids_two": input_ids_two,
            }

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )


        unet_lora_config = LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.rank,
            inference_mode=False,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        model = get_peft_model(self.unet, unet_lora_config)

        # training_args = TrainingArguments(
        #     output_dir=f"./checkpoints/diffusers/adapters/LoRAs/{folder}/",
        #     per_device_train_batch_size=self.args.train_batch_size,
        #     per_device_eval_batch_size=self.args.train_batch_size,
        #     num_train_epochs=self.args.num_train_epochs,
        #     eval_strategy="steps",
        #     save_steps=self.args.checkpointing_steps,
        #     save_total_limit=self.args.checkpoints_total_limit,
        #     logging_dir=f"./outputs/logs/{folder}",
        # )
        # trainer = Trainer(model=model,
        #                   args=training_args,
        #                   train_dataset=train_dataset,
        #                   eval_dataset=eval_dataset,
        #                   data_collator=collate_fn)
        # trainer.train()
        # print()
        # print()
        # print("YAY!")
        # print()
        # print()

        model.train()

        optimizer = AdamW(model.parameters(), lr=5e-5)
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.args.max_train_steps
        )

        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % self.args.logging_steps == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

                if step % self.args.checkpointing_steps == 0:
                    model.save_pretrained(f"./model_checkpoint_{step}")