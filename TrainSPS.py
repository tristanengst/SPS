import argparse
from tqdm import tqdm
import wandb

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sentence_transformers import util as sentence_transformer_utils

from Data import *
from Evaluation import classification_eval
from utils.UtilsContrastive import *
from utils.Utils import *
from utils.UtilsNN import *

class NTXEntLoss(nn.Module):
    """NT-XEnt loss, modified from PyTorch Lightning."""

    def __init__(self, temp=.5):
        """Args:
        temp    -- contrastive loss temperature
        """
        super(NTXEntLoss, self).__init__()
        self.temp = temp

    def forward(self, fx1, fx2):
        """Returns the loss from pre-normalized projections [fx1] and [fx2]."""
        out = torch.cat([fx1, fx2], dim=0)
        n_samples = len(out)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temp)
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        pos = torch.exp(torch.sum(fx1 * fx2, dim=-1) / self.temp)
        pos = torch.cat([pos, pos], dim=0)
        return -torch.log(pos / neg).mean()

class SPSLoss(nn.Module):
    """NT-XEnt loss, modified from PyTorch Lightning."""

    def __init__(self, temp=.5):
        """Args:
        temp    -- contrastive loss temperature
        """
        super(SPSLoss, self).__init__()
        self.temp = temp

    def forward(self, fx1, fx2, C):
        """
        fx1 -- 
        fx2 --
        C   --
        """
        out = torch.cat([fx1, fx2], dim=0)
        n_samples = len(out)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temp)
        mask = ~torch.eye(n_samples, device=sim.device).bool()

        ########################################################################
        # Differences from normal NT-Xent start here
        #
        # Each element of [neg] corresponds to a negative pair. Each element of
        # C represents the semantic similarity between two images, on a scale of
        # 1 / e to e. Therefore 1 / C gives the semantic dissimilarity of the
        # pairs. We can reweight neg by (1 / C), forcing the network to pay more
        # attention to the semantically dissimilar pairs.
        ########################################################################
        neg = sim.masked_select(mask).view(n_samples, -1)

        C = torch.exp(C / self.temp)
        C_inverse = 1 / C
        C_inverse = C_inverse / C_inverse.sum()

        C_inverse = torch.tile(C, (2,2))
        C_inverse = C_inverse.masked_select(mask).view(n_samples, -1)

        neg_times_text_sim = torch.multiply(neg, C_inverse)
        neg_times_text_sim = neg_times_text_sim * (neg.sum() / neg_times_text_sim.sum())
        neg_times_text_sim = neg_times_text_sim.sum(dim=-1)
        
        pos = torch.exp(torch.sum(fx1 * fx2, dim=-1) / self.temp)
        pos = torch.cat([pos, pos], dim=0)
        return -torch.log(pos / neg_times_text_sim).mean()

def one_epoch_sps(model, optimizer, loader, scheduler, temp=.5):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch.

    The returned loss is averaged over batches.

    model           -- a model of the form projection_head(feature_extractor())
    optimizer       -- the optimizer for [model]
    loader          -- a DataLoader over the data to train on
    temp            -- contrastive loss temperature
    """
    model.train()
    loss_fn = SPSLoss(temp)
    loss_total = 0
    scaler = GradScaler()

    for (x1,x2),c in tqdm(loader, desc="Batches", total=len(loader), leave=False, dynamic_ncols=True):

        c = c.to(device, non_blocking=True)
        C = sentence_transformer_utils.cos_sim(c, c)

        with autocast():
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x1.float().to(device, non_blocking=True)),
                           model(x2.float().to(device, non_blocking=True)),
                           C)

        scaler.scale(loss.unsqueeze(0)).backward()
        scaler.step(optimizer)
        loss_total += loss.item()
        scaler.update()
        scheduler.step()

    return model, optimizer, scheduler, loss_total / len(loader)

def one_epoch_basic(model, optimizer, loader, scheduler, temp=.5):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch.

    The returned loss is averaged over batches.

    model           -- a model of the form projection_head(feature_extractor())
    optimizer       -- the optimizer for [model]
    loader          -- a DataLoader over the data to train on
    temp            -- contrastive loss temperature
    """
    model.train()
    loss_fn = NTXEntLoss(temp)
    loss_total = 0
    scaler = GradScaler()

    for x1,x2 in tqdm(loader, desc="Batches", total=len(loader), leave=False, dynamic_ncols=True):

        with autocast():
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x1.float().to(device, non_blocking=True)),
                           model(x2.float().to(device, non_blocking=True)))

        scaler.scale(loss.unsqueeze(0)).backward()
        scaler.step(optimizer)
        loss_total += loss.item()
        scaler.update()
        scheduler.step()

    return model, optimizer, scheduler, loss_total / len(loader)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--wandb", default=1, choices=[0, 1], type=int,
        help="Whether to use W&B logging or not")
    P.add_argument("--data_path", default=f"{project_dir}/data", type=str,
        help="path to data if not in normal place")
    P.add_argument("--data", required=True,
        default="cifar10",
        help="dataset to load images from")
    P.add_argument("--resume", default=None, type=str,
        help="file to resume from")
    P.add_argument("--suffix", default="", type=str,
        help="suffix")

    # Non-hyperparameter arguments
    P.add_argument("--num_workers", default=24, type=int,
        help="Number of workers for data loading")
    P.add_argument("--eval_iter", default=10, type=int,
        help="number of epochs between linear evaluations")
    P.add_argument("--save_iter", default=100, type=int,
        help="save a model every --save_iter epochs")
    P.add_argument("--unreal_augs", default=1, type=int, choices=[0, 1],
        help="whether to use augs that can take an image off the real manifold")
    P.add_argument("--sps_loss", default=0, type=int, choices=[0, 1],
        help="whether to use the proposed loss function or not")
    P.add_argument("--num_augs", default=19, type=int,
        help="whether to use the proposed loss function or not")

    # Hyperparameter arguments
    P.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"],
        help="Resnet backbone to use")

    P.add_argument("--color_s", default=1, type=float,
        help="color distortion strength")
    P.add_argument("--gaussian_blur", choices=[0, 1], type=int, default=1,
        help="include Gaussian blur in data augmentation")
    P.add_argument("--crop_size", type=int, default=128,
        help="resolution at which to feed images to the network")

    P.add_argument("--bs", default=1000, type=int,
        help="batch size")
    P.add_argument("--epochs", default=1000, type=int,
        help="number of epochs")
    P.add_argument("--lars", default=1, choices=[0, 1],
        help="whether or not to use LARS")
    P.add_argument("--lr", default=1e-3, type=float,
        help="base learning rate")
    P.add_argument("--mm", nargs="+", default=(.9, .999), type=float,
        help="momentum (one arg for SGD, two—beta1 and beta2 for Adam)")
    P.add_argument("--n_ramp", default=10, type=int,
        help="Number of linear ramp epochs at start of training")
    P.add_argument("--proj_dim", default=128, type=int,
        help="dimension of projection space")
    P.add_argument("--temp", default=.5, type=float,
        help="contrastive loss temperature")
    P.add_argument("--trust", default=.001, type=float,
        help="LARS trust coefficient")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")
    args = P.parse_args()

    ############################################################################
    # Check arguments
    ############################################################################
    if not args.save_iter % args.eval_iter == 0:
        tqdm.write("WARNING: training will save a checkpoint without direct evaluation. Ensure --save_iter % --eval_iter is zero to avoid this.")
    if args.data in no_val_split_datasets and args.eval == "val":
        raise ValueError("The requested dataset has no validation split. Run with --eval test or cv instead.")

    ############################################################################
    # Load prior state if it exists, otherwise instantiate a new training run.
    ############################################################################
    if args.resume is not None:
        run_id, resume_data = wandb_load(args.resume)
        cur_seed = set_seed(resume_data["seed"])
        data_path = args.data_path

        wandb.init(id=run_id, resume="must", project="sps")
        wandb.save("*.pt")
        model = resume_data["model"].to(device)
        optimizer = resume_data["optimizer"]
        last_epoch = resume_data["last_epoch"]
        args = resume_data["args"]
        args.data_path = data_path

        save_dir = sps_folder(args)
    else:
        cur_seed = set_seed(args.seed)
        args.run_id = wandb.util.generate_id()
        save_dir = sps_folder(args)

        wandb.init(anonymous="allow", id=args.run_id, project="sps",
            mode="online" if args.wandb else "disabled", config=args,
            name=f"{args.data}-{args.backbone}-num_augs{args.num_augs}sps_loss{args.sps_loss}-unreal_augs{args.unreal_augs}{args.run_id}{suffix_str(args)}")

        model = HeadedResNet(args.backbone, args.proj_dim,
            head_type="projection",
            small_image=(args.data in small_image_datasets))
        model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
        optimizer = Adam(get_param_groups(model, args.lars), lr=args.lr,
            betas=args.mm, weight_decay=1e-6)
        optimizer = LARS(optimizer, args.trust) if args.lars else optimizer
        last_epoch = -1

    tqdm.write(dict_to_nice_str(vars(args)))

    ############################################################################
    # Instantiate the scheduler and get the data
    ############################################################################
    data_tr, data_eval = get_data_splits(args.data, args.sps_loss,
        data_path=args.data_path, num_augs=args.num_augs)

    if args.unreal_augs:
        augs_tr, augs_fn, augs_te = get_contrastive_augs(color_s=args.color_s,
            crop_size=args.crop_size, gaussian_blur=args.gaussian_blur)
    else:
        augs_tr, augs_fn, augs_te = get_real_augs(crop_size=args.crop_size)

    if args.sps_loss:
        data_ssl = CaptionAndGeneratedImagesDataset(args.data,
            data_path=args.data_path,
            image_transform=augs_tr,
            return_embeddings=True,
            num_samples=2)
    else:
        data_ssl = ManyTransformsDataset(data_tr, augs_tr, augs_tr)

    loader = DataLoader(data_ssl, shuffle=True, batch_size=args.bs,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
        **seed_kwargs(cur_seed))
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=args.epochs * len(loader), max_lr=args.lr, 
        min_lr=1e-6, warmup_steps=args.n_ramp * len(loader),
        last_epoch=last_epoch if last_epoch == -1 else last_epoch * len(loader))

    tqdm.write(f"Dataset length {len(data_tr)}")

    one_epoch = one_epoch_sps if args.sps_loss else one_epoch_basic

    ############################################################################
    # Begin training!
    ############################################################################
    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1), desc="Epochs", dynamic_ncols=True):
        model, optimizer, scheduler, loss_tr = one_epoch(model,
            optimizer, loader, scheduler, args.temp)

        ########################################################################
        # LOG RESULTS OF THE EPOCH. Perform a classification cross validation if
        # desired, and otherwise print/log results or merely that the epoch
        # happened
        ########################################################################
        if e % args.eval_iter == 0 and not e == 0 and args.eval_iter > 0:

            # Extract the backbone from the model. This is gross :(
            if hasattr(model, "backbone"):
                backbone = model.backbone
            elif hasattr(model, "module"):
                backbone = model.module.backbone
            else:
                raise ValueError(f"Got model of unknown type '{type(model)}")

            val_acc_avg, val_acc_std = classification_eval(backbone, data_eval,
                "cv", augs_fn, augs_te, trials=3, data_name=args.data,
                data_path=args.data_path, split="val")

            wandb.log({"epoch": e, "loss_tr": loss_tr / len(loader),
                "acc_val": val_acc_avg, "lr": scheduler.get_lr()[0]})
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_lr()[0]:.5f} | loss_tr {loss_tr / len(loader):.5f} | val acc {val_acc_avg:.5f} ± {val_acc_std:.5f}")
        else:
            wandb.log({"epoch": e, "loss_tr": loss_tr / len(loader),
                "lr": scheduler.get_lr()[0]})
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_lr()[0]:.5f} | loss_tr {loss_tr / len(loader):.5f}")

        if e % args.save_iter == 0 and not e == 0:
            wandb_save({"model": model, "optimizer": optimizer, "args": args,
                "last_epoch": e}, f"{sps_folder(args)}/{e}.pt")
            tqdm.write("Saved training state")

        
