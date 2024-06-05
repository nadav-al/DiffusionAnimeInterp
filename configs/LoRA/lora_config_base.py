# LoRA params
rank = 4
t_modules = ["to_k", "to_q", "to_v", "to_out.0"]

# Optimizer params
lr = 1e-05
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_w_decay = 1e-2
adam_epsilon = 1e-08

checkpointing_steps = 250
