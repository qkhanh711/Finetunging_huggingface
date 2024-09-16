import os
import glob
from PIL import Image
import json
import subprocess
import torch

dataset_name = "TS_class0"
local_dir = "./00000"
# from ppm2png import save_ppm_to_png
# save_ppm_to_png("./", local_dir.split("/")[-1])

import locale
locale.getpreferredencoding = lambda: "UTF-8"
subprocess.run(['accelerate', 'config', 'default'])

# from huggingface_hub import notebook_login
# notebook_login()

# Launch training with DreamBooth and LoRA
import os
import dotenv
from huggingface_hub import login

dotenv.load_dotenv()
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=TOKEN)

subprocess.run([
    'accelerate', 'launch', 'diffusers/train_dreambooth_lora_sdxl.py',
    '--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"',
    '--pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"',
    '--dataset_name="Nyanmaru/BelgiumTS_class0"',
    '--output_dir="TS_LoRA"',
    '--caption_column="prompt"',
    '--mixed_precision="fp16"',
    '--instance_prompt="a photo of TOK 00000"',
    '--resolution=1024',
    '--train_batch_size=1',
    '--gradient_accumulation_steps=3',
    '--gradient_checkpointing',
    '--learning_rate=1e-4',
    '--snr_gamma=5.0',
    '--lr_scheduler="constant"',
    '--lr_warmup_steps=0',
    '--use_8bit_adam',
    '--max_train_steps=50',
    '--checkpointing_steps=717',
    '--seed="0"'
])

from huggingface_hub import whoami, create_repo, upload_folder
from train_dreambooth_lora_sdxl import save_model_card
from pathlib import Path

output_dir = "00000_LoRA"
username = whoami(token=Path("/root/.cache/huggingface/"))["name"]
repo_id = f"{username}/{output_dir}"
print(f"Your model is available at https://huggingface.co/{repo_id}")

repo_id = create_repo(repo_id, exist_ok=True).repo_id

save_model_card(
    repo_id=repo_id,
    images=[],
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    train_text_encoder=False,
    instance_prompt="a photo of TOK traffic sign, 00000 with ",
    validation_prompt=None,
    repo_folder=output_dir,
    vae_path="madebyollin/sdxl-vae-fp16-fix",
    use_dora=False,
)

upload_folder(
    repo_id=repo_id,
    folder_path=output_dir,
    commit_message="End of training",
    ignore_patterns=["step_*", "epoch_*"]
)

# Display link to model
from IPython.display import display, Markdown

link_to_model = f"https://huggingface.co/{repo_id}"
display(Markdown(f"### Your model has finished training.\nAccess it here: {link_to_model}"))

# Inference with the trained model
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.load_lora_weights(repo_id)
_ = pipe.to("cuda")

prompt = "a photo of TOK messi eating chocolate"
image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image
image.save("sdxl.png")
