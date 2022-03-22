import argparse
import os
import shutil
from tqdm import tqdm

def create_files(image_file, text_file, data_dir):
    """
    """
    with open(text_file, "r") as f:
        captions = f.read().split("\n")

    for idx,c in enumerate(captions):
        text_output_file = f"{data_dir}/{os.path.basename(text_file).replace('.txt', f'_{idx}.txt')}"
        image_output_file = f"{data_dir}/{os.path.basename(image_file).replace('.jpg', f'_{idx}.jpg')}"

        with open(text_output_file, "w+") as f:
            f.write(c)

        shutil.copy(image_file, image_output_file)

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--image_dir")
    P.add_argument("--caption_dir")
    P.add_argument("--output_dir")
    args = P.parse_args()

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    image_classes = sorted(os.listdir(args.image_dir))
    text_classes = sorted(os.listdir(args.caption_dir))

    if not image_classes == text_classes:
        raise ValueError("Got different classes")

    for img_cls,txt_cls in tqdm(zip(image_classes, text_classes), total=len(image_classes)):
        images = sorted(os.listdir(f"{args.image_dir}/{img_cls}"))
        captions = sorted(os.listdir(f"{args.caption_dir}/{txt_cls}"))

        if not [img[:-4] for img in images] == [cap[:-4] for cap in captions]:
            print(images, captions, img_cls, txt_cls)
            raise ValueError("Got different A")

        for img,cap in zip(images, captions):
            # create_files(
            #     f"{args.image_dir}/{img_cls}/{img}",
            #     f"{args.caption_dir}/{txt_cls}/{cap}",
            #     args.output_dir,
            # )
            text_output_file = f"{args.output_dir}/{cap}"
            image_output_file = f"{args.output_dir}/{img}"
            shutil.copy(f"{args.image_dir}/{img_cls}/{img}", image_output_file)
            shutil.copy(f"{args.caption_dir}/{txt_cls}/{cap}", text_output_file)
