import os
import subprocess
import locale
import dotenv
import wandb
from huggingface_hub import whoami, upload_folder, create_repo
from diffuserss.train_dreambooth_lora_flux import save_model_card
from pathlib import Path
from IPython.display import display, Markdown

# Set locale to UTF-8
locale.getpreferredencoding = lambda: "UTF-8"

# Load environment variables
dotenv.load_dotenv()
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
KEY = os.getenv("WANDB_API_KEY")

# Configure Accelerate
subprocess.run(["accelerate", "configure", "default"])

# Login to Wandb
wandb.login(key=KEY)

# Define command parameters
command = [
    "accelerate", "launch",
    "diffusers/train_dreambooth_lora_flux.py",
    "--pretrained_model_name_or_path=black-forest-labs/FLUX.1-dev",
    "--instance_data_dir=00000",
    "--output_dir=LoRA_flux",
    "--mixed_precision=bf16",
    "--instance_prompt= a photo of TOK traffic sign, 00000 with ocean waves in the background",
    "--resolution=512",
    "--train_batch_size=1",
    "--guidance_scale=1",
    "--gradient_accumulation_steps=4",
    "--optimizer=prodigy",
    "--learning_rate=1.",
    "--report_to=wandb",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--max_train_steps=50",
    "--validation_prompt= a photo of TOK traffic sign, 00000 with ocean waves in the background",
    "--validation_epochs=25",
    "--seed=0",
    "--wandb_project=LoRA_flux",
]

# Run the training command
subprocess.run(command)

# Set the output directory and Hugging Face repo details
output_dir = "00000_LoRA_sd3"
username = whoami(token=Path("/root/.cache/huggingface/"))["name"]
repo_id = f"{username}/{output_dir}"
print(f"Your model is available at https://huggingface.co/{repo_id}")

# Create a Hugging Face repository if it doesn't exist
repo_id = create_repo(repo_id, exist_ok=True).repo_id

# Save the model card
save_model_card(
    repo_id=repo_id,
    images=[],
    base_model="black-forest-labs/FLUX.1-dev",
    train_text_encoder=False,
    instance_prompt="a photo of TOK traffic sign, 00000 with ",
    validation_prompt=None,
    repo_folder=output_dir,
    vae_path=None,
    use_dora=False,
)

# Upload the trained model to Hugging Face
upload_folder(
    repo_id=repo_id,
    folder_path=output_dir,
    commit_message="End of training",
    ignore_patterns=["step_*", "epoch_*"],
)

# Display link to the trained model
link_to_model = f"https://huggingface.co/{repo_id}"
display(Markdown(f"### Your model has finished training.\nAccess it here: {link_to_model}"))

# Inference with the trained model

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe = pipe.to("cuda")
pipe.load_lora_weights(repo_id)

prompt = "a photo of TOK traffic sign, 00000 with ocean waves in the background"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux.png")


