import argparse

import DALLE
from Data import TextDataset
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from utils.Utils import *

quote_str = "\n\t"

def generate_one_epoch(model, loader, epoch_number, save_path):
    """Generates images for one epoch.

    Args:
    model           -- a function mapping from a batch of captions to a batch of
                        images
    loader          -- a DataLoader over (index,caption) pairs
    epoch_number    -- the number of the epoch having images generated for it
    save_path       -- path to save images to. Generated images are saved to
                        save_path/index/image1_augepoch_number.jpg
    """
    for idxs,captions in tqdm(loader, desc="Generating one epoch", leave=False, dynamic_ncols=True):
        images = model(captions)
        for idx,image in zip(idxs, images):
            save_image(image, f"{save_path}/{idx}/image1_aug{epoch_number}.jpg")

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--data", required=True,
        help="dataset name")
    P.add_argument("--data_path", type=str, default=f"{project_dir}/data",
        help="Path to datasets directory")
    P.add_argument("--split", choices=["train", "val", "test"], default="train",
        help="split of data to get augmentations for")
    P.add_argument("--gpu", type=int, default=0,
        help="ID of GPU to use")
    P.add_argument("--start_epoch", type=int, default=0,
        help="Index of epoch to start generating at")
    P.add_argument("--epochs", type=int, default=1,
        help="Number of epochs to generate data for")
    P.add_argument("--bs", type=int, default=1,
        help="ID of GPU to use")
    P.add_argument("--start_stop_idxs", type=int, required=True, nargs=2,
        help="ID of GPU to use")
    P.add_argument("--name", default=None, type=str,
        help="Name of dataset to generate")
    args = P.parse_args()

    # Get the captions and wrap a DataLoader over it
    data = TextDataset(f"{args.data}/{args.split}", data_path=args.data_path,
        return_idxs=True)
    data = Subset(data, range(*args.start_stop_idxs))
    tqdm.write(f"Found dataset containing {len(data)} captions. The first few:\n{quote_str + quote_str.join([data[i][1] for i in range(0,5)])}\n")
    loader = DataLoader(data, batch_size=args.bs, shuffle=False,
        num_workers=min(args.bs, 8))
    model = DALLE.get_generate_images_fn(f"cuda:{args.gpu}")

    # Create folders to save images to. Each folder should contain a .txt file
    # containing the caption it goes with.
    args.name = f"gen_{args.data}" if args.name is None else args.name
    save_path = f"{args.data_path}/{args.name}/{args.split}"
    for idx,caption in tqdm(data, desc="Building data folder structure", dynamic_ncols=True):
        path = f"{save_path}/{idx}"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/caption.txt", "w+") as f:
            f.write(caption)

    # Generate images
    for e in tqdm(range(args.start_epoch, args.epochs + args.start_epoch), desc="Generating epochs", dynamic_ncols=True):
        generate_one_epoch(model, loader, e, save_path)
