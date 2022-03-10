"""Script for creating a data folders for training DALL-E. Given a TSV file
containing (image URL, cpation) pairs and a name of the dataset to be generated,
creates the following folder structure inside a data folder:

📂 data
 ┣ 📂 dataset_name_images
 ┃  ┣ 📜 cat.png
 ┃  ┗ ...
 ┣ 📂image-and-text-data
 ┃  ┣ 📜cat.png
 ┃  ┣ 📜cat.txt
 ┃  ┗ ...
"""
import argparse
import json
import os
import random
import shutil
from tqdm import tqdm

from Utils import *

if __name__ == "__main__":
    P = argparse.ArgumentParser("")
    P.add_argument("--data_dir", type=str, default=data_dir,
        help="Directory for storing project data. Only override if nondefault")
    P.add_argument("--annotations", required=True, type=str,
        help="COCO data annoations JSON")
    P.add_argument("--images", required=True, type=str,
        help="Folder containing images")
    P.add_argument("--num_val", type=int, default=50,
        help="Number of images taken for validation")
    P.add_argument("--seed", type=int, default=0,
        help="Random seed for dataset splitting")
    args = P.parse_args()

    data_dir = args.data_dir

    # Create folders to put outputs in
    image_output_dir = f"{data_dir}/coco_images"
    image_caption_output_dir = f"{data_dir}/coco_captions_images"
    if not os.path.exists(f"{image_output_dir}/train"):
        os.makedirs(f"{image_output_dir}/train")
    if not os.path.exists(f"{image_output_dir}/val"):
        os.makedirs(f"{image_output_dir}/val")
    if not os.path.exists(f"{image_caption_output_dir}/train"):
        os.makedirs(f"{image_caption_output_dir}/train")
    if not os.path.exists(f"{image_caption_output_dir}/val"):
        os.makedirs(f"{image_caption_output_dir}/val")

    # Load the annotations JSON and build dictionaries we will use to map things
    # together so DALL-E can read them
    with open(args.annotations, "r") as f:
        data = json.load(f)
    id2file = {x["id"]: x["file_name"] for x in data["images"]}
    id2file = {id: f"{args.images}/{file}" for id,file in id2file.items()}
    id2caption = {x["image_id"]: x["caption"] for x in data["annotations"]}
    id_file_caption = [(id, id2file[id], c) for id,c in id2caption.items()]

    # Split the data
    random.seed(args.seed)
    val_idxs = set(random.sample(range(len(id_file_caption)), k=args.num_val))

    # Create and manipulate files
    for idx,(id,file,caption) in tqdm(enumerate(id_file_caption), total=len(id_file_caption)):
        name = f"val/{id}" if idx in val_idxs else f"train/{id}"

        with open(f"{image_caption_output_dir}/{name}.txt", "w+") as f:
            f.write(caption)
        shutil.copy(file, f"{image_caption_output_dir}/{name}.jpg")
        shutil.copy(file, f"{image_output_dir}/{name}.jpg")
