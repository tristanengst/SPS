import argparse
from collections import defaultdict
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from Data import *
from utils.UtilsContrastive import *
from utils.Utils import *
from torch.cuda.amp import GradScaler, autocast

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def accuracy(model, loader):
    """Returns the accuracy of [model] on [loader]."""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            preds = torch.argmax(model(x.to(device)), dim=1)
            correct += torch.sum((preds == y.to(device))).item()
            total += len(preds)

    return correct / total

def get_y2idxs(data_tr, data_name=None, data_path=f"{project_dir}/data", split=None):
    """Returns a label to indices mapping for [data_tr]. Together, [data_name]
    and [split] can be used to memoize the mapping.
    
    Args:
    data_tr     -- a dataset returning XY pairs
    data_name   -- the name of the dataset
    split  -- the split of the dataset
    """
    # If the mapping already exists, return it
    if data_name is not None and split is not None:
        save_path = f"{data_path}/{data_name}/{split}_label2idx.json"
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                return json.load(f)

    # Build the mapping
    y2idxs = defaultdict(lambda: [])
    for idx,(_,y) in tqdm(enumerate(data_tr), desc="Building y2idxs", total=len(data_tr)):
        y2idxs[y].append(idx)

    # Memoize the mapping if possible
    if data_name is not None and split is not None:
        if not os.path.exists(f"{data_path}/{data_name}/{split}_label2idx.json"):
            with open(f"{data_path}/{data_name}/{split}_label2idx.json", "w+") as f:
                json.dump(y2idxs, f)

    return y2idxs



def get_eval_data(data_tr, data_te, augs_fn, augs_te, F, precompute_feats=True, num_workers=24, bs=1000):
    """Returns validation training and testing datasets. The testing dataset may
    be a string 'cv' to indicate cross validation if [data_te] is 'cv'.

    Args:
    data_tr -- training data
    data_te -- testing data
    augs_fn -- training augmentations
    augs_te -- testing augmentations
    F       -- a HeadlessResNet for constructing features only once or None for
                doing it on the fly with different augmentations
    precompute_feats    -- whether to precompute features.
    """
    if precompute_feats:
        return (FeatureDataset(XYDataset(data_tr, augs_fn), F, bs=bs,num_workers=num_workers),
                FeatureDataset(XYDataset(data_te, augs_te), F, bs=bs, num_workers=num_workers))
    else:
        return XYDataset(data_tr, augs_fn), XYDataset(data_te, augs_te)

def get_eval_trial_accuracy(data_tr, data_te, out_dim, num_classes, trial=0, epochs=100, bs=64, num_workers=24, F=None):
    """Returns the accuracy of a linear model trained on features from [F] of
    [data_tr] on [data_te].

    Args:
    data_tr     -- the data for training the linear model
    data_te     -- the data for evaluating the accuracy of the linear model
    F           -- a feature extractor (HeadlessResNet) or None if [data_tr] and
                    [data_te] already consist of features
    epochs      -- the number of epochs to train for
    bs          -- the batch size to use
    """
    # Check inputs
    if (len(data_tr[0][0].shape) > 1 or len(data_te[0][0].shape) > 1) and F is None:
        raise ValueError("If data is not a single-order tensor, feature extractor F must not be None")
    
    # If [F] is None, convert it to an idenity function to pretend to use it
    F = nn.Identity() if F is None else F

    # Construct training and testing utilities
    loader_tr = DataLoader(data_tr, shuffle=True, pin_memory=True,
        batch_size=bs, drop_last=False, num_workers=num_workers,
        **seed_kwargs(trial))
    loader_te = DataLoader(data_te, pin_memory=True, batch_size=1024,
        drop_last=False, num_workers=num_workers)
    model = nn.Linear(out_dim, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=epochs * len(loader_tr), max_lr=5e-3, min_lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="mean").to(device)
    scaler = GradScaler()

    for e in tqdm(range(epochs), desc="Validation epochs", leave=False):

        for x,y in loader_tr:
            with autocast():
                with torch.no_grad():
                    x = F(x.to(device, non_blocking=True))

                model.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    return accuracy(nn.Sequential(F, model), loader_te)


from sklearn.model_selection import train_test_split

def classification_eval(feature_extractor, data_tr, data_te, augs_fn, augs_te,
    precompute_feats=True, ex_per_class="all", trials=3, epochs=100, bs=64,
    num_workers=24, data_path=None, data_name=None, split=None):
    """Returns evaluation accuracy of feature extractor [feature_extractor].

    Args:
    feature_extractor   -- feature extractor
    data                -- data to test generalization to
    augs_fn             -- augmentations for training (of linear model) data
    precompute_feats    -- precompute features. This is massively faster because
                            the CNN need be run only once, but reduces the
                            number of augmentations used in total.
    ex_per_class        -- 'all' or a number. Gives the number of examples per
                            class that should be used in training. (This would
                            be useful for few-shot learning.)
    trials              -- number of trials. If [data_te] is 'cv', number of
                            cross-validation folds
    """
    def get_split_idxs(idxs, split_index, include):
        """Computes a split of [idxs] to be used for cross-validation.

        Args:
        idxs        -- the indices to be split
        split_index -- the index of the split (which trial is being run)
        include     -- whether to return indices between the computed start and
                        endpoints in [idxs], or those exluding those indices
        """
        start = split_index * (len(idxs) // trials)
        end = (1 + split_index) * (len(idxs) // trials)
        return idxs[start:end] if include else idxs[:start] + idxs[end:]

    y2idxs = get_y2idxs(data_tr, data_path=data_path, data_name=data_name, split=split)
    out_dim = feature_extractor.out_dim
    num_classes = len(y2idxs)
    accuracies = []

    for t in tqdm(range(trials), desc="Validation trials"):

        if data_te == "cv":
            y2idxs_tr = {y: get_split_idxs(y2idxs[y], t, include=False) for y in y2idxs}
            y2idxs_te = {y: get_split_idxs(y2idxs[y], t, include=True) for y in y2idxs}
            
            if not ex_per_class == "all":
                idxs_tr = [random.sample(y2idxs_tr[y], ex_per_class) for y in y2idxs_tr]
            else:
                idxs_tr = [idx for y in y2idxs_tr for idx in y2idxs_tr[y]]

            idxs_te = [idx for y in y2idxs_te for idx in y2idxs_te[y]]
            trial_data_tr = Subset(data_tr, indices=flatten(idxs_tr))
            trial_data_te = Subset(data_tr, indices=flatten(idxs_te))
        else:
            if not ex_per_class == "all":
                idxs_tr = [random.sample(y2idxs[y], ex_per_class) for y in y2idxs]
            else:
                idxs_tr = [idx for y in y2idxs for idx in y2idxs[y]]
            trial_data_tr = Subset(data_tr, indices=flatten(idxs_tr))
            trial_data_te = data_te

        trial_data_tr, trial_data_te = get_eval_data(trial_data_tr, trial_data_te,
            augs_fn, augs_te, F=feature_extractor,
            precompute_feats=precompute_feats, num_workers=num_workers, bs=8 * bs)
        acc = get_eval_trial_accuracy(trial_data_tr, trial_data_te, out_dim,
            num_classes, epochs=epochs, bs=bs, trial=t, num_workers=num_workers,
            F=None if precompute_feats else feature_extractor)
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies) * 1.96 / np.sqrt(trials)