import argparse
import math
import os

from datasets import Dataset, Features, Image
from torch.utils.data import DataLoader
# import torch_2_1_1 as torch
# import torch_2_3
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
# import transformers
# import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
# from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import _set_state_dict_into_text_encoder, compute_snr
# from diffusers.training_utils import cast_training_params
# from diffusers.utils import (
#     check_min_version,
#     convert_state_dict_to_diffusers,
#     convert_unet_state_dict_to_peft,
#     is_wandb_available,
# )
from peft import LoraConfig, set_peft_model_state_dict
# from peft import LoraModel, LoraConfig
# import peft
from PIL import Image
from tqdm.auto import tqdm

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
        default=1024,
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
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




class LoraTrainer:
    def __init__(self, config, args=None):
        self.config = config
        self.args = parse_args(args)
    def __prepare_dataloader(self, input, type="tensors"):
        device = torch.device("cuda")
        trans = TF.Compose([TF.ToTensor()])
        if type == "str":
            images = [trans(Image.open(img[0])) for img in input]
        elif type == "images":
            images = [trans(img) for img in input]
        elif type == "tensors":
            images = input

        images = images * self.args.image_repeats

        ds = Dataset.from_dict({'image': images})
        return DataLoader(ds, shuffle=True, batch_size=self.args.train_batch_size,
                          num_workers=self.args.dataloader_num_workers)

    def __train(self, dataloader, folder):
        args = self.args
        noise_scheduler = DDPMScheduler.from_pretrained(self.config.diff_path, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(self.config.diff_path, subfolder="unet", revision=args.revision)
        unet.requires_grad_(False)
        unet.to("cuda", dtype=args.weight_dtype) # TODO weight_dtype?

        vae = AutoencoderKL.from_pretrained(self.config.diff_path, subfolder="vae",
                                            revision=args.revision, variant=args.variant)
        vae.requires_gard_(False)
        vae.to("cuda", dtype=args.weight_dtype) # TODO weight_dtype?

        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)

        ### Notice that here you should probably add the inner functions in lines 638-724 on train_text_to_image_lora_sdxl.py

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate = (
                    args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size
            )

        # # Make sure the trainable params are in float32.
        # if args.mixed_precision == "fp16":
        #     models = [unet]
        #     cast_training_params(models, dtype=torch.float32)

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )
        print("Training!")
        global_step = 0
        for epoch in range(args.num_train_epochs):
            progress_bar = tqdm(total=len(dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                # Convert images to latent space
                latents = vae.encode(batch["image"].to(dtype=args.weight_dtype)).latent_dist.sample() # TODO weight_dtype?
                latents = latents * vae.config.scaling_factor
                noise = torch.randn(latents.shape, device=latents.device)
                bs = latents.shape[0]
                # Sample a random timestep for each image

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (bs,), device=latents.device, dtype=torch.int64)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(noisy_latents, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1
            if self.config.to_save:
                if (epoch + 1) % self.config.save_models_epochs == 0 or epoch == self.config.num_epochs - 1:
                    pipeline = StableDiffusionXLPipeline(unet=unet, scheduler=noise_scheduler)
                    pipeline.save_pretrained(self.config.ckpt_path + folder)

        print("Done LoRA Training on ", folder)
        return unet


    def train_from_tensors(self, images, folder):
        dataloader = self.__prepare_dataloader(images)
        return self.__train(dataloader, folder)

    def train_from_images(self, path, folder):
        dataloader = self.__prepare_dataloader(path)
        return self.__train(dataloader, folder)
