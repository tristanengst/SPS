import argparse
import lpips
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_transformer_utils
import wandb

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Data import *
from Models import *
from Utils import *

class SPSLoss(nn.Module):
    """Returns SPS Loss."""
    def __init__(self, temp=.5):
        super(SPSLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.temp = temp
        self.reduction = "mean"

    def forward(self, x1, x2, y):
        bs, d = x1.shape
        dists = self.cosine_similarity(x1.repeat(bs, 1), x2.repeat(bs, 1))
        dists = torch.exp(dists / self.temp).view(bs, bs)
        numerator = 1 - (torch.matmul(dists, torch.eye(bs, device=device)) - y)
        denomenator = torch.sum(dists)
        loss = torch.mean(-1 * torch.log(torch.divide(numerator, denomenator)))
        return loss

def get_loss_fn(lpips_frac=0):
    """Returns a convex combination of SPS and LPIPs loss with the LPIPS loss
    weighted to be [lpips_frac] of the total.
    """
    if lpips_frac > 0:
        raise NotImplementedError()
    else:
        return SPSLoss()

text_sim_model = None
def get_text_distances(sentences):
    """Returns a Tensor of all pairwise cosine distances among [sentences]."""
    global text_sim_model
    if text_sim_model is None:
        text_sim_model = SentenceTransformer("bert-base-nli-mean-tokens", device=device)

    embeddings = text_sim_model.encode(sentences, convert_to_tensor=True,
        device=device, normalize_embeddings=True)
    return sentence_transformer_utils.dot_score(embeddings, embeddings)

def one_epoch(dalle, model, optimizer, loader, scheduler,
    grad_norm=float("inf"), num_prints=10, **kwargs):
    """Returns a (model, optimizer, loss_tr) tuple after training [model] for
    one epoch.

    Args:
    dalle       -- pretrained DALLE model
    model       -- model being trained
    optimizer   -- optimizer for [model]
    loader      -- DataLoader over a TextDataset
    scheduler   -- learning rate scheduler
    grad_norm   -- value to clip gradient norms to
    num_prints  -- number of times to print during the epoch
    """
    total_loss = 0
    scaler = GradScaler()

    for idx,(sentences,tokens) in tqdm(loader, desc="Batches", leave=False, dynamic_ncols=True):
        with autocast():
            with torch.no_grad():
                distances = get_text_distance(sentences)
                images = dalle(tokens)

            loss = loss_fn(model(x1), model(x2), distances)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if idx % (len(loader) // num_prints) == 0:
            tqdm.write(f"\tcurr_loss {loss} | lr {scheduler.get_last_lr()[0]}")

        scheduler.step()

    return model, optimizer, total_loss / len(loader)


if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SPS training")
    P.add_argument("--data_dir", type=str, default=f"{project_dir}/data",
        help="Directory to find data folders int")
    P.add_argument("--wandb", choices=[0, 1], default=1, type=int,
        help="Enable WandB")
    P.add_argument("--data", required=True, type=str,
        help="prefix of data folder. --data_dir/--data should be a valid path to it")
    P.add_argument("--dalle", required=True, type=str,
        help="Path to load DALLE from")
    P.add_argument("--transform", choices=["basic"],
        help="Text transformation")
    P.add_argument("--save_iter", default=10, type=int,
        help="Number of iterations between model saves")
    P.add_argument("--resume", default=None, type=str,
        help="checkpoint to resume from")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")
    P.add_argument("--arch", default="vgg_linear_rescale", choices=["vgg_linear_rescale"],
        help="random seed")
    P.add_argument("--suffix", default=None, type=str,
        help="checkpoint to resume from")

    P.add_argument("--num_workers", default=24, type=int,
        help="Number of dataloader workers")

    P.add_argument("--bs", type=int, default=1,
        help="Batch size")
    P.add_argument("--epochs", type=int, default=1,
        help="Number of training epochs")
    P.add_argument("--grad_norm", type=float, default=float("inf"),
        help="Batch size")
    P.add_argument("--lr", type=float, default=1e-3,
        help="Base learning rate")

    args = P.parse_args()

    if args.resume is not None:
        run_id, resume_data = wandb_load(args.resume)
        cur_seed = set_seed(resume_data["seed"])
        data_folder_path = args.data_folder_path

        wandb.init(id=run_id, resume="must", project="sps-metric")
        wandb.save("*.pt")
        model = resume_data["model"].to(device)
        optimizer = resume_data["optimizer"]
        last_epoch = resume_data["last_epoch"]
        args = resume_data["args"]
        args.data_folder_path = data_folder_path

        save_dir = sps_folder(args)
    else:
        cur_seed = set_seed(args.seed)
        args.run_id = wandb.util.generate_id()
        save_dir = sps_folder(args)

        wandb.init(anonymous="allow", id=args.run_id, project="sps-metric",
            mode="online" if args.wandb else "disabled", config=args,
            name=f"{args.data}_{args.arch}_{args.run_id}{suffix_str(args)}")

        last_epoch = -1
        dalle = load_dalle(args.dalle)
        model = nn.DataParallel(SPSModel(**vars(args)), device_ids=args.gpus).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)

    tqdm.write(dict_to_nice_str(vars(args)))

    scheduler = CosineAnnealingWarmRestarts(optimizer,
        (len(loader) * args.epochs // 16), eta_min=1e-6, last_epoch=last_epoch)

    if args.transform == "basic":
        transform = "basic"
    else:
        raise ValueError(f"Unknown transform")
    data = TextDataset(args.data, transform)
    loader = DataLoader(data, shuffle=True, batch_size=args.bs,
        num_workers=args.num_workers)

    for e in tqdm(range(args.epochs), desc="Epochs",  dynamic_ncols=True):
        model, optimizer, loss_tr = one_epoch(dalle, model, optimizer, loader,
            scheduler, **vars(args))

        tqdm.write(f"End of epoch {e} | loss_tr {loss_tr:.5e} | loss_val {loss_val:.5e}")
        wandb.log({"epoch": e, "loss_tr": loss_tr, "lr": scheduler.get_lr()[0]})

        if e % args.save_iter == 0 and not e == 0:
            wandb_save({"model": model, "optimizer": optimizer, "args": args,
                "last_epoch": e}, f"{simclr_folder(args)}/{e}.pt")
            tqdm.write("Saved training state")
