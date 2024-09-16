---
base_model: stabilityai/stable-diffusion-xl-base-1.0
library_name: diffusers
license: openrail++
tags:
- text-to-image
- diffusers-training
- diffusers
- lora
- template:sd-lora
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- text-to-image
- diffusers-training
- diffusers
- lora
- template:sd-lora
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
instance_prompt: a photo of TOK messi
widget: []
---

<!-- This model card has been generated automatically according to the information the training script had access to. You
should probably proofread and complete it, then remove this comment. -->


# SDXL LoRA DreamBooth - Nyanmaru/messi_LoRA

<Gallery />

## Model description

These are Nyanmaru/messi_LoRA LoRA adaption weights for stabilityai/stable-diffusion-xl-base-1.0.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: False.

Special VAE used for training: madebyollin/sdxl-vae-fp16-fix.

## Trigger words

You should use a photo of TOK messi to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download](Nyanmaru/messi_LoRA/tree/main) them in the Files & versions tab.



## Intended uses & limitations

#### How to use

```python
# TODO: add an example code snippet for running this diffusion pipeline
```

#### Limitations and bias

[TODO: provide examples of latent issues and potential remediations]

## Training details

[TODO: describe the data used to train the model]