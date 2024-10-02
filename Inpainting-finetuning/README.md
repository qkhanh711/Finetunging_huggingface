# **DATASET**   
    kaggle datasets download -d sebastiendelprat/inpaintingdataset

It takes 24G VRAM for infrerence (Not able to finetune)

# Test Images

    
    wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
    
    wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png

# **Diffusers**

    wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/controlnet/train_controlnet_sd3.py (not run)
    wget https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/controlnet_sd3.py (runable but not enough VRAM)
    

