import argparse

import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.optim import Adam, CosineAnnealingWarmRestarts


class SPSModel(nn.Module):

    def __init__(self):
        super(SPSModel, self).__init__()
        # For now, just a ResNet-18
        model = models.resnet18(pretrained=False)
        self.model = nn.Sequential(
            *[l for n,l in model.named_children() if not n == "fc"])

    def forward(self, x): return self.model(x)

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
    if lpips_frac > 0:
        return
    else:
        return SPSLoss()


text_distance_model = None
def get_text_distances(x, y):
    """
    """
    global text_distance_model
    if text_distance_model is None:
        text_distance_model = None


    return distances

def one_epoch(dalle, model, optimizer, loader, text_augmenter, scheduler,
    logger, grad_norm=2, num_prints=10, **kwargs):
    """
    """
    total_loss = 0
    scaler = GradScaler()
    for idx,(s1,s2) in tqdm(loader, desc="Batches", leave=False, dynamic_ncols=True):

        with autocast():
            with torch.no_grad():
                distances = get_text_distance(s1, s2)
                x1, x2 = dalle(s1, s2)

            loss = loss_fn(model(x1), model(x2), distances)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(dalle.parameters(), grad_norm)
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
    P.add_argument("--wandb", choices=[0, 1], default=1,
        help="Enable WandB")
    P.add_argument("--data", choices=[0, 1], default=1,
        help="Enable WandB")
    P.add_argument("--dalle", required=True, type=str,
        help="Path to load DALLE from")

    P.add_argument("--bs", type=int, default=1,
        help="Batch size")
    P.add_argument("--epochs", type=int, default=1,
        help="Number of training epochs")
    P.add_argument("--grad_norm", type=float, default=2,
        help="Batch size")
    P.add_argument("--lr", type=float, default=1e-3,
        help="Base learning rate")

    P.add_argument("--options", default=[], nargs="+")
    args = P.parse_args()

    args.options = sorted(args.options + [
        f"bs{args.bs}"
        f"epochs{args.epochs}"
        f"lr{args.lr}"
    ])

    last_epoch = -1
    dalle = load_dalle(args.dalle)
    model = nn.DataParallel(SPSModel(**vars(args)), device_ids=args.gpus).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
        (len(loader) * args.epochs // 16), eta_min=1e-6, last_epoch=last_epoch)

    for e in tqdm(range(args.epochs), desc="Epochs",  dynamic_ncols=True):
        model, optimizer, loss_tr = one_epoch(dalle, model, optimizer, loader,
            text_augmenter)

        tqdm.write(f"End of epoch {e} | loss_tr {loss_tr:.5e} | loss_val {loss_val:.5e}")
