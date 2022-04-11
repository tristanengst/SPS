import os

"""Code for building datasets.

See data/SetupDataset.py for downloading their underlying data.

The important portions of this file are get_data_splits(), which returns
training and evaluation ImageFolders, and the various Dataset subclasses that
can be used to construct various useful datasets.
"""
from collections import OrderedDict, defaultdict
import numpy as np
import random
import sys
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split

from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torchvision.transforms.functional import hflip
from torchvision.datasets.folder import default_loader

from utils.Utils import *

################################################################################
# Dataset medatata. In general, unless it's properly logged here, you can use a
# new dataset. If it is logged here, using it should be easy!
#
# To make running on a dataset possible, ensure images in it can be accessed via
# `data/dataset_name/split/class/image.png` where `dataset_name` doesn't include
# resolution information, ie. an actual path would be
#
#   `data/cifar10_16x16/train/airplane/image.png`
#
# but only `cifar10` would be recorded here.
################################################################################
def dataset_has_no_captions(data_str):
    """Returns if a dataset specified by [data_str] has no captions."""
    return "miniImagenet" in data_str

no_val_split_datasets = ["cifar10"]
small_image_datasets = ["cifar10"]
data2split2n_class = {
    "cifar10": {"train": 10, "val": 10, "test": 10},
    "miniImagenet": {"train": 64, "val": 16, "test": 20}
}

def seed_kwargs(seed=0):
    """Returns kwargs to be passed into a DataLoader to give it seed [seed]."""
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {"generator": g, "worker_init_fn": seed_worker}

################################################################################
# Functionality for loading datasets
################################################################################

def get_data_splits(data_str, return_captions, data_path=f"{project_dir}/data",
    num_augs=float("inf")):
    """Returns data for training and evaluation. All Datasets returned are
    ImageFolders, meaning that another kind of dataset likely needs to be built
    on top of them.

    Args:
    data_str    -- string specifying the dataset to load. It must exactly exist
                    in the data directory, ie. data/data_str exists.
    eval_str    -- how to get validation/testing data
    data_path   -- path to dataset; data can be found at data_path/data_str
    """
    if "miniImagenet" in data_str and not return_captions:
        data_tr = PreAugmentedImageFolder(f"{data_path}/{data_str}/train", num_augs=num_augs)
        data_val = ImageFolder(f"{data_path}/miniImagenet/val")
    elif "miniImagenet" in data_str and return_captions:
        raise ValueError("Can not return captions with miniImagenet")
    elif "gen_coco" in data_str and not return_captions:
        data_tr = PreAugmentedImageFolder(f"{data_path}/{data_str}/train", num_augs=num_augs)
        data_val = ImageFolder(f"{data_path}/miniImagenet/val")
        tqdm.write("NOTE: Validation data is miniImagenet val split")
    elif "gen_coco" in data_str  and return_captions:
        data_tr = PreAugmentedImageFolder(f"{data_path}/{data_str}/train", num_augs=num_augs)
        data_val = ImageFolder(f"{data_path}/miniImagenet/val")
        tqdm.write("NOTE: Validation data is miniImagenet val split")
    elif "coco" in data_str  and not return_captions:
        data_tr = PreAugmentedImageFolder(f"{data_path}/{data_str}/train", num_augs=num_augs)
        data_val = ImageFolder(f"{data_path}/miniImagenet/val")
        tqdm.write("NOTE: Validation data is miniImagenet val split")
    elif "coco" in data_str  and return_captions:
        data_tr = PreAugmentedImageFolder(f"{data_path}/{data_str}/train", num_augs=num_augs)
        data_val = ImageFolder(f"{data_path}/miniImagenet/val")
        tqdm.write("NOTE: Validation data is miniImagenet val split")
    
    else:
        raise ValueError(f"Unknown dataset '{data_str}")

    return data_tr, data_val

################################################################################
# Augmentations
################################################################################
def get_real_augs(crop_size=32):
    """Returns augmentations that ensure images remain on the real manifold."""
    augs_tr = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    augs_te = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor()])
    return augs_tr, augs_tr, augs_te

def get_contrastive_augs(crop_size=32, gaussian_blur=False, color_s=0):
    """Returns a (SSL transforms, finetuning transforms, testing transforms)
    tuple based on [data_str].

    Args:
    data_str  -- a string specifying the dataset to get transforms for
    color_s   -- the strength of color distortion
    strong    -- whether to use strong augmentations or not
    """
    color_jitter = transforms.ColorJitter(0.8 * color_s,
         0.8 * color_s, 0.8 * color_s, 0.2 * color_s)
    color_distortion = transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2)])

    augs_tr = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        color_distortion,
        transforms.GaussianBlur(23, sigma=(.1, 2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    augs_te = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor()])

    return augs_tr, augs_tr, augs_te

################################################################################
# Datasets
################################################################################

class XYDataset(Dataset):
    """A simple dataset returning examples of the form (transform(x), y)."""

    def __init__(self, data, transform=transforms.ToTensor(), memoize=False):
        """Args:
        data        -- a sequence of (x,y) pairs
        transform   -- the transform to apply to each returned x-value
        """
        super(XYDataset, self).__init__()
        self.transform = transform
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return self.transform(image), label

class FeatureDataset(Dataset):
    """A dataset of model features.

    Args:
    F       -- a feature extractor, eg. the backbone of a ResNet
    data    -- a dataset of XY pairs
    bs      -- the batch size to use for feature extraction
    """

    def __init__(self, data, F, bs=1000, num_workers=24):
        super(FeatureDataset, self).__init__()
        loader = DataLoader(data, batch_size=bs, drop_last=False, pin_memory=True, num_workers=num_workers)

        data_x, data_y = [], []
        F = F.to(device)
        F.eval()
        with torch.no_grad():
            for x,y in tqdm(loader, desc="Building FeatureDataset", leave=False, file=sys.stdout):
                data_x.append(F(x.to(device)).cpu())
                data_y.append(y)

        data_x = [x for x_batch in data_x for x in x_batch]
        data_y = [y for y_batch in data_y for y in y_batch]
        self.data = list(zip(data_x, data_y))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]


class ManyTransformsDataset(Dataset):
    """A dataset that wraps a single source dataset, but returns a tuple of
    transformed items from it, with the ith item coming from the ith transform
    in [transforms].

    Args:
    source_dataset  -- the source dataset
    *transforms     -- the transforms to use
    """
    def __init__(self, source_dataset, *transforms):
        super(ManyTransformsDataset, self).__init__()
        self.source_dataset = source_dataset
        self.transforms = transforms

    def __len__(self): return len(self.source_dataset)

    def __getitem__(self, idx):
        x = self.source_dataset[idx][0]
        return tuple([t(x) for t in self.transforms])

class PreAugmentedImageFolder(Dataset):
    """A drop-in replacement for an ImageFolder for use where some
    augmentations are pre-generated. It will behave differently as described
    below.

    Args:
    source              -- path to an folder of images laid out for an
                            ImageFolder. Files which differ by only an `_augN`
                            string are considered augmentations of each other.
    transform           -- transform to apply
    target_transform    -- target transform
    num_augs            -- the maximum number of augmentations that can be
                            returned for each image        
    verbose             -- whether to print info about constructed dataset
    """
    def __init__(self, source, transform=None, target_transform=None,
        verbose=True, num_augs=float("inf")):

        def remove_aug_info(s):
            """Returns string [s] without information indicating which
            augmentation it is. Concretely, this means that the `_augN` where
            `N` is some (possibly multi-digit) number substring is removed.

            This requires images to be named without breaking this function.
            """
            if "_aug" in s:
                underscore_idx = s.find("_")
                dot_idx = s.find(".")
                return f"{s[:underscore_idx]}{s[dot_idx]}"
            else:
                return s

        # Build a mapping from keys representing unique images to indices to the
        # files under [source] that are augmentations of that image
        image2idxs, counter = defaultdict(lambda: []), 0
        for c in tqdm(os.listdir(source), leave=False, desc="Buidling PreAugmentedImageFolder"):
            for image in os.listdir(f"{source}/{c}"):
                if os.path.splitext(image)[1].lower() in [".jpg", ".jpeg", ".png"]:
                    image2idxs[remove_aug_info(f"{c}/{image}")].append(counter)
                    counter += 1
        
        image2idxs = {img: idxs[:min(len(idxs), num_augs)] for img,idxs in image2idxs.items()}

        super(PreAugmentedImageFolder, self).__init__()
        self.data_idx2aug_idxs = [v for v in image2idxs.values() if len(v) > 0]
        self.data = ImageFolder(source, transform=transform,
            target_transform=target_transform)
        self.num_classes = len(os.listdir(source))

        # Print dataset statistics
        if verbose:
            aug_stats = [len(idxs) for idxs in self.data_idx2aug_idxs]
            s = f"Constructed PreAugmentedImageFolder over {source}. Length: {len(self.data_idx2aug_idxs)} | Min augmentations for an image: {min(aug_stats)} | Average: {np.mean(aug_stats):.5f}| Max: {max(aug_stats)}"
            tqdm.write(s)


    def __len__(self): return len(self.data_idx2aug_idxs)

    def __getitem__(self, idx):
        return self.data[random.choice(self.data_idx2aug_idxs[idx])]

def is_image_file(f):
    f = f.lower()
    return f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")


class CaptionAndGeneratedImagesDataset(Dataset):
    """A dataset returning a (caption, [images]) tuple.

    Args:
    source            -- a dataset under [data_path] organized as follows:

                            source
                            |------- folder1
                            |       |------- caption.txt (optional)
                            |       |------- caption_embedding.pt (optional)
                            |       |------- generated_image_1.jpg
                            |       |------- ...
                            |       |------- generated_image_N.jpg
                            |------- ...
    num_samples         -- number of images to return along with a caption
    data_path           -- path to datasets
    return_embeddings   -- whether to return embeddings of captions
    """

    def __init__(self, source, split="train", num_samples=1, data_path=f"{project_dir}/data", return_embeddings=False, image_transform=transforms.ToTensor()):
        super(CaptionAndGeneratedImagesDataset, self).__init__()

        self.return_embeddings = return_embeddings
        self.num_samples = num_samples
        self.image_transform = image_transform
        self.data = []
        for folder in os.listdir(f"{data_path}/{source}/{split}"):
            
            folder = f"{data_path}/{source}/{split}/{folder}"
            if os.path.isfile(folder):
                tqdm.write(f"Skipping file {folder}")
            
            caption = None
            images = []

            for file in os.listdir(folder):
                if is_image_file(file):
                    images.append(f"{folder}/{file}")
                elif file.endswith(".txt") and not self.return_embeddings:
                    with open(f"{folder}/{file}", "r") as f:
                        caption = f.read().strip().lower()
                elif file.endswith("embedding.pt") and self.return_embeddings:
                    caption = torch.load(f"{folder}/{file}")
                elif file.endswith(".txt") and self.return_embeddings:
                    continue
                elif file.endswith("embedding.pt") and not self.return_embeddings:
                    continue
                else:
                    raise ValueError(f"Got unknown file {file}")

            if len(images) == 0:
                raise ValueError(f"Found not images under {folder}")
            if caption is None:
                raise ValueError(f"Found no caption under {folder}")
            self.data.append((caption, images))

        self.loader = default_loader

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        caption, image_files = self.data[idx]
        image_files = random.choices(image_files, k=self.num_samples)
        images = [self.image_transform(self.loader(f)) for f in image_files]
        return images, caption

def print_batch(b, indent=""):
    s = ""
    if isinstance(b, (list, tuple)):
        inner = [print_batch(b_, indent=indent + "    ") for b_ in b]
        s += f"{indent}[\n" + "\n".join(inner) + f"\n{indent}]"
    elif isinstance(b, str):
        s += f"{indent}string: {b}"
    elif isinstance(b, torch.Tensor):
        s += f"{indent}tensor of size {b.shape}"
    return s

class TextDataset(Dataset):
    """A dataset returning strings and optionally indices of strings.
    Args:
    source      -- a dataset under [data_path] organized as follows:

                    source
                    |------- folder1
                    |       |------- caption.txt
                    |       |------- generated_image_1.jpg
                    |       |------- ...
                    |       |------- generated_image_N.jpg
                    |------- ...

                    but where the images are optional
    num_samples -- number of images to return along with a caption
    data_path   -- path to datasets
    return_idxs -- whether to return (idx, string) or just a string
    """
    def __init__(self, source, data_path=f"{project_dir}/data", return_idxs=False):
        super(TextDataset, self).__init__()
        source = f"{data_path}/{source}"

        self.data = []
        for folder in sorted(os.listdir(source)):
            if folder.endswith(".txt"):
                file = folder
                if file.endswith(".txt"):
                     with open(f"{source}/{file}", "r") as f:
                        self.data.append(f.read().strip().lower())
            elif os.path.isdir(folder):
                for file in sorted(os.listdir(f"{source}/{folder}")):
                    if file.endswith(".txt"):
                         with open(f"{source}/{folder}/{file}", "r") as f:
                            self.data.append(f.read().strip().lower())
            else:
                continue

        self.return_idxs = return_idxs

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return (idx, self.data[idx]) if self.return_idxs else self.data[idx]
