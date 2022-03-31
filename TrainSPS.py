"""

GPU USAGE:
- I assume two GPUs
- DALLE should be DataParallel on both GPUs
- CLIP should be on GPU 1
- the SPS model should be on GPU 0
"""
import argparse
from functools import partial
import lpips
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_transformer_utils
import wandb

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from Data import *
from Models import *
from Utils import *

# minDALLE imports
# from dalle.models import Dalle
# from dalle.utils.utils import set_seed, clip_score
# generate_images_objects = None
# def generate_image(text, top_k=200, num_candidates=32):
#     """Returns an image given [text].

#     Args:
#     dalle           -- a DALLE model. Handle CUDAFication somewhere else.
#     clip            -- a (CLIP, preprocess_clip, preprocess_clip_device) tuple.
#                         The CLIP model
#                         should be moved to a device somewhere else
#     text            -- the text (string) to generate images from
#     top_k           -- integer, higher is better but uses more memory
#     num_candidates  -- more is better but takes more time
#     """
#     # If DALLE and CLIP haven't been loaded, load them.
#     if generate_images_objects is None:
#         dalle = Dalle.from_pretrained('minDALL-E/1.3B').to("cuda:0")
#         clip_model, preprocess_clip = clip.load("ViT-B/32", device="cuda:1")
#         clip_model.to("cuda:1")
#         generate_images_objects = (dalle, clip, preprocess_clip)
#     else:
#         dalle, clip, preprocess_clip = generate_images_objects

#     images = dalle.sampling(prompt=text, top_k=top_k,
#         num_candidates=num_candidates)
#     images = np.transpose(images, (0, 2, 3, 1))
#     rank = clip_score(prompt=text, images=images, model_clip=clip,
#         preprocess_clip=preprocess_clip, device="cuda:1")
#     return images[rank[0]]


# GLIDE immports
# from glide_text2im.download import load_checkpoint
# from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults, model_and_diffusion_defaults_upsampler
#
# glide, diffusion = None, None
# def internal_glide_fn(x_t, ts, guidance_scale=3, glide=glide, **kwargs):
#     """Function needed to make GLIDE work."""
#     half = x_t[: len(x_t) // 2]
#     combined = torch.cat([half, half], dim=0)
#     glide_out = glide(combined, ts, **kwargs)
#     eps, rest = glide_out[:, :3], glide_out[:, 3:]
#     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#     half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
#     eps = torch.cat([half_eps, half_eps], dim=0)
#     return torch.cat([eps, rest], dim=1)
#
# def generate_image(text, guidance_scale=3, upsample_temp=.997, glide_device="cuda:1"):
#     global glide
#     global diffusion
#
#     options = model_and_diffusion_defaults()
#     options["use_fp16"] = True
#     if glide is None:
#         glide, diffusion = create_model_and_diffusion(**options)
#         glide.to(glide_device)
#         glide.load_state_dict(load_checkpoint('base', glide_device))
#
#     print(text)
#
#     tokens = glide.tokenizer.encode(text)
#     tokens, mask = glide.tokenizer.padded_tokens_and_mask(tokens,
#         options['text_ctx'])
#
#     batch_size = 1
#     uncond_tokens, uncond_mask = glide.tokenizer.padded_tokens_and_mask(
#         [], options['text_ctx'])
#     glide_kwargs = dict(
#     tokens=torch.tensor(
#         [tokens] * batch_size + [uncond_tokens] * batch_size, device=glide_device
#     ),
#     mask=torch.tensor(
#         [mask] * batch_size + [uncond_mask] * batch_size,
#         dtype=torch.bool,
#         device=glide_device,
#     ),
# )
#
#     glide.del_cache()
#     samples = diffusion.p_sample_loop(
#         partial(internal_glide_fn, glide=glide, guidance_scale=guidance_scale),
#         (batch_size, 3, options["image_size"], options["image_size"]),
#         device=glide_device,
#         clip_denoised=True,
#         progress=False,
#         model_kwargs=glide_kwargs,
#         cond_fn=None)[:batch_size]
#     glide.del_cache()
#
#     return samples

from GLIDE import glide_generate

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

def one_epoch(sps_model, optimizer, loader, scheduler, grad_norm=float("inf"),
    num_prints=10, **kwargs):
    """Returns a (model, optimizer, loss_tr) tuple after training [model] for
    one epoch.

    Args:
    model       -- model being trained
    optimizer   -- optimizer for [model]
    loader      -- DataLoader over a TextDataset
    scheduler   -- learning rate scheduler
    grad_norm   -- value to clip gradient norms to
    num_prints  -- number of times to print during the epoch
    """
    total_loss = 0
    scaler = GradScaler()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6).to("cuda:0")

    for idx,sentences in tqdm(enumerate(loader), total=len(loader), desc="Batches", leave=False, dynamic_ncols=True):
        with autocast():
            with torch.no_grad():
                images = glide_generate(sentences).to("cuda:0")
                text_distances = get_text_distances(sentences)

            fx = sps_model(images)

            fx1 = fx.unsqueeze(0).expand((fx.shape[0],) + fx.shape).reshape((fx.shape[0] ** 2,) + fx.shape[1:])
            fx2 = fx.repeat_interleave(fx.shape[0], dim=0)


            image_distances = cos(fx1, fx2).view(fx.shape[0], fx.shape[0])





            loss = torch.mean((image_distances - text_distances) ** 2)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # clip_grad_norm_(sps_model.parameters(), grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        tqdm.write(f"loss {loss.item()}")

        if idx % (len(loader) // num_prints) == 0:
            tqdm.write(f"\tcurr_loss {loss} | lr {scheduler.get_last_lr()[0]}")

        scheduler.step()

    return sps_model, optimizer, total_loss / len(loader)


if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SPS training")
    P.add_argument("--data_dir", type=str, default=f"{project_dir}/data",
        help="Directory to find data folders int")
    P.add_argument("--wandb", choices=[0, 1], default=1, type=int,
        help="Enable WandB")
    P.add_argument("--data", required=True, type=str,
        help="prefix of data folder. --data_dir/--data should be a valid path to it")
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

    ############################################################################
    # INITIALIZATION STEP. Either instantiate a new run-specific traiining
    # objects or load old ones.
    ############################################################################
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
        sps_model = ScaledVGGFeatures().to("cuda:0")
        # model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = AdamW(sps_model.parameters(), lr=args.lr, weight_decay=1e-8)

    tqdm.write(dict_to_nice_str(vars(args)))

    dataset = HumanAugmentationTextDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.bs, num_workers=0, shuffle=False, collate_fn=collate_fn)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
        (len(loader) * args.epochs // 16), eta_min=1e-6, last_epoch=last_epoch)

    for e in tqdm(range(args.epochs), desc="Epochs",  dynamic_ncols=True):
        sps_model, optimizer, loss_tr = one_epoch(sps_model, optimizer, loader,
            scheduler, **vars(args))

        tqdm.write(f"End of epoch {e} | loss_tr {loss_tr:.5e} | loss_val {loss_val:.5e}")
        wandb.log({"epoch": e, "loss_tr": loss_tr, "lr": scheduler.get_lr()[0]})

        if e % args.save_iter == 0 and not e == 0:
            wandb_save({"model": model, "optimizer": optimizer, "args": args,
                "last_epoch": e}, f"{simclr_folder(args)}/{e}.pt")
            tqdm.write("Saved training state")
