from transformers import Trainer
import torch.nn as nn
import torch
from typing import Dict, Union, Any
class InterpTrainer(Trainer):
    def __init__(self, loss_func=nn.L1Loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Overrides the training_step method to perform training with LoRA.

        Args:
          eval_phase (bool, optional): Whether in evaluation mode (default: False).

        Returns:
          torch.Tensor: Calculated training loss.
        """
        # Forward pass, calculate loss and perform backpropagation
        loss = self.compute_loss(model, inputs)

        # Return loss for logging
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        clean_img = inputs[0]
        distorted_img = inputs[1]
        outputs = model(distorted_img).images[0]
        loss = self.loss_func(clean_img, outputs)
        return (loss, outputs) if return_outputs else loss