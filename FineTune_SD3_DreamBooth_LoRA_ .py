#!/usr/bin/env python
# coding: utf-8

# Imports
from PIL import Image
import glob
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import locale
import os
import dotenv
import json
import wandb
from huggingface_hub import whoami, login, notebook_login, create_repo, upload_folder
from train_dreambooth_lora_sdxl import save_model_card
from IPython.display import display, Markdown
import subprocess
from pathlib import Path

# Set locale encoding to UTF-8
locale.getpreferredencoding = lambda: "UTF-8"

# Preview images from local directory
local_dir = "./00000/"  # Directory containing images

# Accelerate config setup
subprocess.run(['accelerate', 'config', 'default'])

# Hugging Face login
dotenv.load_dotenv()
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=TOKEN)

# WandB login
KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=KEY)

# Launch training
subprocess.run([
    'accelerate', 'launch', 'diffusers/train_dreambooth_lora_sd3.py',
    '--pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers"',
    '--instance_data_dir="00000"',
    '--output_dir="TS_LoRA_sd3"',
    '--mixed_precision="bf16"',
    '--instance_prompt="a photo of TOK 00000"',
    '--resolution=512',
    '--train_batch_size=4',
    '--gradient_accumulation_steps=4',
    '--learning_rate=0.0001',
    '--report_to="wandb"',
    '--lr_scheduler="constant"',
    '--lr_warmup_steps=0',
    '--max_train_steps=50',
    '--weighting_scheme="logit_normal"',
    '--seed="42"',
    '--use_8bit_adam',
    '--gradient_checkpointing',
    '--prior_generation_precision="bf16"'
])

# Save model to Hugging Face repository
output_dir = "00000_LoRA_sd3"
username = whoami(token=Path("/root/.cache/huggingface/"))["name"]
repo_id = f"{username}/{output_dir}"
print(f"Your model is available at https://huggingface.co/{repo_id}")

repo_id = create_repo(repo_id, exist_ok=True).repo_id
save_model_card(
    repo_id=repo_id,
    images=[],
    base_model="stabilityai/stable-diffusion-3-medium-diffusers",
    train_text_encoder=False,
    instance_prompt="a photo of TOK traffic sign, 00000",
    validation_prompt=None,
    repo_folder=output_dir,
    vae_path="madebyollin/taesd3",
    use_dora=False,
)

upload_folder(
    repo_id=repo_id,
    folder_path=output_dir,
    commit_message="End of training",
    ignore_patterns=["step_*", "epoch_*"]
)

# Display link to the trained model
link_to_model = f"https://huggingface.co/{repo_id}"
display(Markdown(f"### Your model has finished training.\nAccess it here: {link_to_model}"))

# Inference with the trained model
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=torch.float16)
pipe.load_lora_weights("TS_LoRA_sd3/pytorch_lora_weights.safetensors")
pipe = pipe.to("cuda")

prompt = "a photo of TOK traffic sign, 00000 with ocean waves in the background"
image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image
image.save("sd3.png")
