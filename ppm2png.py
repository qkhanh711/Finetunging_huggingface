import os
import glob

from PIL import Image
import json
dataset_name = "TS_class0"
local_dir = "./00000"


def save_ppm_to_png(root, dir):
    try:
        os.remove(f"{root}/{dir}/*.csv")
        os.remove(f"{root}/{dir}/*.jsonl")
    except:
        pass
    for ppm in glob.glob(f"{root}/{dir}/*.ppm"):
        img = Image.open(ppm)
        img.save(ppm.replace(".ppm", ".png"))
        os.remove(ppm)
save_ppm_to_png("./", local_dir.split("/")[-1])