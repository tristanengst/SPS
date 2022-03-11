import argparse
from pathlib import Path
import time
from glob import glob
import os
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE, DiscreteVAE, DALLE
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer

# libraries needed for webdataset support
import webdataset as wds
from torchvision import transforms as T
from PIL import Image
from io import BytesIO

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

parser=argparse.ArgumentParser()
parser.add_argument("--wandb", choices=[0, 1], default=1, type=int,
    help="use wandb or not")
parser.add_argument("--gpus", type=int, nargs="+", default=[0],
    help="GPUS to use")
group=parser.add_mutually_exclusive_group(required=False)
group.add_argument("--vae_path", type=str,
    help="path to your trained discrete VAE")
group.add_argument("--dalle_path", type=str,
    help="path to your partially trained DALL-E")
parser.add_argument("--vqgan_model_path", type=str, default=None,
    help="path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)")
parser.add_argument("--vqgan_config_path", type=str, default=None,
    help="path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)")
parser.add_argument("--image_text_folder", type=str, required=True,
    help="path to your folder of images and text for learning the DALL-E")
parser.add_argument("--wds", type=str, default="",
    help="Comma separated list of WebDataset (1) image and (2) text column names. Must contain 2 values, e.g. img,cap.")
parser.add_argument("--truncate_captions", dest="truncate_captions", action="store_true",
    help="Captions passed in which exceed the max token length will be truncated if this is set.")
parser.add_argument("--random_resize_crop_lower_ratio", dest="resize_ratio", type=float, default=0.75,
    help="Random resized crop lower ratio")
parser.add_argument("--chinese", dest="chinese", action="store_true")
parser.add_argument("--taming", dest="taming", action="store_true")
parser.add_argument("--hug", dest="hug", action="store_true")
parser.add_argument("--bpe_path", type=str,
    help="path to your BPE json file")
parser.add_argument("--dalle_output_file_name", type=str, default="dalle",
    help="output_file_name")
parser.add_argument("--fp16", action="store_true",
    help="(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.")
parser.add_argument("--amp", action="store_true",
	help="Apex 'O1' automatic mixed precision. More stable than 16 bit precision. Can\"t be used in conjunction with deepspeed zero stages 1-3.")
parser.add_argument("--wandb_name", default="dalle_train_transformer",
    help="Name W&B will use when saving results.\ne.g. `--wandb_name `coco2017-full-sparse`")
parser.add_argument("--wandb_entity", default=None,
    help="(optional) Name of W&B team/entity to log to.")
parser.add_argument("--stable_softmax", dest="stable_softmax", action="store_true",
    help="Prevent values from becoming too large during softmax. Helps with stability in fp16 and Mixture of Quantization training.")

train_group=parser.add_argument_group("Training settings")

train_group.add_argument("--flops_profiler", dest="flops_profiler", action="store_true",
    help="Exits after printing detailed flops/runtime analysis of forward/backward")
train_group.add_argument("--epochs", default=20, type=int,
    help="Number of epochs")
train_group.add_argument("--save_every_n_steps", default=1000, type=int,
    help="Save a checkpoint every n steps")
train_group.add_argument("--keep_n_checkpoints", default=None, type=int,
    help="(Careful) Deletes old deepspeed checkpoints if there are more than n")
train_group.add_argument("--batch_size", default=4, type=int,
    help="Batch size")
train_group.add_argument("--ga_steps", default=1, type=int,
    help="Number of steps to accumulate gradients across per each iteration. DeepSpeed only.")
train_group.add_argument("--learning_rate", default=3e-4, type=float,
    help="Learning rate")
train_group.add_argument("--clip_grad_norm", default=0.5, type=float,
    help="Clip gradient norm")
train_group.add_argument("--lr_decay", dest="lr_decay", action="store_true",
    help="enable learning rate decay")

model_group=parser.add_argument_group("Model settings")
model_group.add_argument("--dim", default=512, type=int,
    help="Model dimension")
model_group.add_argument("--text_seq_len", default=256, type=int,
    help="Text sequence length")
model_group.add_argument("--depth", default=2, type=int,
    help="Model depth")
model_group.add_argument("--heads", default=8, type=int,
    help="Model number of heads")
model_group.add_argument("--dim_head", default=64, type=int,
    help="Model head dimension")
train_group.add_argument("--ff_dropout", default=0.0, type=float,
    help="Feed forward dropout.")
train_group.add_argument("--attn_dropout", default=0.0, type=float,
    help="Feed forward dropout.")
model_group.add_argument("--reversible", dest="reversible", action="store_true")
model_group.add_argument("--loss_img_weight", default=7, type=int,
        help="Image loss weight")
model_group.add_argument("--attn_types", default="full", type=str,
    help="comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.")
model_group.add_argument("--shift_tokens", help="Use the shift tokens feature", action="store_true")
model_group.add_argument("--rotary_emb", help="Use rotary embeddings", action="store_true")
model_group.add_argument("--shared_attn_ids", default=None, type=str,
    help="Comma separated list of shared attention layer ids. Default: sharing is disabled")
model_group.add_argument("--shared_ff_ids", default=None, type=str,
    help="Comma separated list of shared feed forward layer ids. Default: sharing is disabled")
model_group.add_argument("--share_input_output_emb", action="store_true",
        help="Share input and output embeddings", )

args=parser.parse_args()

################################################################################
# Utility methods and constants
################################################################################
def exists(val): return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def get_pkg_version():
    from pkg_resources import get_distribution
    return get_distribution("dalle_pytorch").version

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path=Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension=cp_path.parent / cp_path.stem
    cp_dir=Path(f"{path_sans_extension}-{tag}-cp")
    return cp_dir

WEBDATASET_IMAGE_TEXT_COLUMNS=tuple(args.wds.split(","))
ENABLE_WEBDATASET=True if len(WEBDATASET_IMAGE_TEXT_COLUMNS) == 2 else False

DALLE_OUTPUT_FILE_NAME=args.dalle_output_file_name + ".pt"

VAE_PATH=args.vae_path
VQGAN_MODEL_PATH=args.vqgan_model_path
VQGAN_CONFIG_PATH=args.vqgan_config_path
DALLE_PATH=args.dalle_path
RESUME=exists(DALLE_PATH)

EPOCHS=args.epochs
BATCH_SIZE=args.batch_size

LEARNING_RATE=args.learning_rate
GRAD_CLIP_NORM=args.clip_grad_norm
LR_DECAY=args.lr_decay
SAVE_EVERY_N_STEPS=args.save_every_n_steps
KEEP_N_CHECKPOINTS=args.keep_n_checkpoints

MODEL_DIM=args.dim
TEXT_SEQ_LEN=args.text_seq_len
DEPTH=args.depth
HEADS=args.heads
DIM_HEAD=args.dim_head
REVERSIBLE=args.reversible
LOSS_IMG_WEIGHT=args.loss_img_weight
FF_DROPOUT=args.ff_dropout
ATTN_DROPOUT=args.attn_dropout
STABLE=args.stable_softmax
SHIFT_TOKENS=args.shift_tokens
ROTARY_EMB=args.rotary_emb

ATTN_TYPES=tuple(args.attn_types.split(","))
SHARED_ATTN_IDS=tuple(args.shared_attn_ids.split(",")) if exists(args.shared_attn_ids) else None
SHARED_FF_IDS=tuple(args.shared_ff_ids.split(",")) if exists(args.shared_ff_ids) else None
SHARE_INPUT_OUTPUT_EMB=args.share_input_output_emb

################################################################################
# Argument checking
################################################################################
assert Path(args.image_text_folder).exists(), f"The path {args.image_text_folder} was not found."

if exists(args.bpe_path):
    klass=HugTokenizer if args.hug else YttmTokenizer
    tokenizer=klass(args.bpe_path)
elif args.chinese:
    tokenizer=ChineseTokenizer()

# reconstitute vae

if RESUME:
    dalle_path=Path(DALLE_PATH)
    assert dalle_path.exists(), "DALL-E model file does not exist"
    loaded_obj=torch.load(str(dalle_path), map_location="cpu")

    dalle_params, vae_params, weights=loaded_obj["hparams"], loaded_obj["vae_params"], loaded_obj["weights"]
    opt_state=loaded_obj.get("opt_state")
    scheduler_state=loaded_obj.get("scheduler_state")

    if vae_params is not None:
        vae=DiscreteVAE(**vae_params)
    elif args.taming:
        vae=VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
    else:
        vae=OpenAIDiscreteVAE()

    IMAGE_SIZE=vae.image_size
    resume_epoch=loaded_obj.get("epoch", 0)
else:
    if exists(VAE_PATH):
        vae_path=Path(VAE_PATH)
        assert vae_path.exists(), "VAE model file does not exist"
        assert not vae_path.is_dir(), \
            ("Cannot load VAE model from directory; please use a "
             "standard *.pt checkpoint. "
             "Currently, merging a DeepSpeed-partitioned VAE into a DALLE "
             "model is not supported.")

        loaded_obj=torch.load(str(vae_path))

        vae_params, weights=loaded_obj["hparams"], loaded_obj["weights"]

        vae=DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        tqdm.write("using pretrained VAE for encoding images to tokens")
        vae_params=None

        if args.taming:
            vae=VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
        else:
            vae=OpenAIDiscreteVAE()

    IMAGE_SIZE=vae.image_size

    dalle_params=dict(
        num_text_tokens=tokenizer.vocab_size,
        text_seq_len=TEXT_SEQ_LEN,
        dim=MODEL_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        reversible=REVERSIBLE,
        loss_img_weight=LOSS_IMG_WEIGHT,
        attn_types=ATTN_TYPES,
        ff_dropout=FF_DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        stable=STABLE,
        shift_tokens=SHIFT_TOKENS,
        rotary_emb=ROTARY_EMB,
        shared_attn_ids=SHARED_ATTN_IDS,
        shared_ff_ids=SHARED_FF_IDS,
        share_input_output_emb=SHARE_INPUT_OUTPUT_EMB,
    )
    resume_epoch=0

# configure OpenAI VAE for float16s

if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
    vae.enc.blocks.output.conv.use_float16=True

# helpers

def group_weight(model):
    group_decay, group_no_decay=[], []
    for params in model.named_parameters():
        if "transformer" in params[0]:
            if "bias" in params[0] or "norm" in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups=[dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


# create dataset and dataloader

imagepreproc=T.Compose([
    T.Lambda(lambda img: img.convert("RGB")
    if img.mode != "RGB" else img),
    T.RandomResizedCrop(IMAGE_SIZE,
                        scale=(args.resize_ratio, 1.),
                        ratio=(1., 1.)),
    T.ToTensor(),
])

def imagetransform(b): return Image.open(BytesIO(b))

def tokenize(s):
    return tokenizer.tokenize(
        s.decode("utf-8"),
        TEXT_SEQ_LEN,
        truncate_text=args.truncate_captions).squeeze(0)

ds = TextImageDataset(
    args.image_text_folder,
    text_len=TEXT_SEQ_LEN,
    image_size=IMAGE_SIZE,
    resize_ratio=args.resize_ratio,
    truncate_captions=args.truncate_captions,
    tokenizer=tokenizer,
    shuffle=False, # DataLoader does this
)
assert len(ds) > 0, "dataset is empty"

tqdm.write(f"{len(ds)} image-text pairs found for training")
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=6)

dalle = nn.DataParallel(DALLE(vae=vae, **dalle_params), device_ids=args.gpus).to(device)

if args.fp16:
    dalle=dalle.half()
dalle=dalle.to("cuda:0")

if RESUME: dalle.load_state_dict(weights)

# optimizerimizer

optimizer=Adam(get_trainable_params(dalle), lr=LEARNING_RATE)

if RESUME and opt_state: optimizer.load_state_dict(opt_state)

# scheduler

scheduler=None

if LR_DECAY:
    scheduler=ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        cooldown=10,
        min_lr=1e-6,
        verbose=True,
    )
    if RESUME and scheduler_state:
        scheduler.load_state_dict(scheduler_state)

# experiment tracker

model_config=dict(
    depth=DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD
)

training_config = dict(
    lr=args.learning_rate,
    lr_decay=args.lr_decay,
    grad_clip_norm=GRAD_CLIP_NORM,
    bs=BATCH_SIZE,
    epochs=EPOCHS,
)

run=wandb.init(
    project=args.wandb_name,
    entity=args.wandb_entity,
    resume=False,
    mode="online" if args.wandb else "disabled",
    config=model_config | training_config,
)

def save_model(path, epoch=0):
    save_obj={
        "hparams": dalle_params,
        "vae_params": vae_params,
        "epoch": epoch,
        "version": get_pkg_version(),
        "vae_class_name": vae.__class__.__name__
    }

    save_obj={
        **save_obj,
        "weights": dalle.state_dict(),
        "opt_state": optimizer.state_dict(),
        "scheduler_state": (scheduler.state_dict() if scheduler else None)
    }

    torch.save(save_obj, path)

def save_artifact(model_config, model_path, name="trained-dalle"):
    model_artifact=wandb.Artifact(name, type="model", metadata=dict(model_config))
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact)

# training

# Saves a checkpoint before training begins to fail early when mis-configured.
# See https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints

save_model(DALLE_OUTPUT_FILE_NAME, epoch=resume_epoch)
scaler = GradScaler()
print_iter = len(dl) // 10
tqdm.write(f"Will print every {print_iter} iterations")
for epoch in tqdm(range(resume_epoch, EPOCHS), desc="Epochs"):

    for i, (text, images) in tqdm(enumerate(dl), desc="Batches", leave=False, total=len(dl)):

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            text, images = map(lambda t: t.to(device), (text, images))
            loss = dalle(text, images, return_loss=True).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(dalle.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        log={}

        if i % print_iter == 0:
            tqdm.write(f"epoch {epoch:3} | iter {i:7} | loss_tr {loss.item()}")

            log={
                **log,
                "epoch": epoch,
                "iter": i,
                "loss": loss.item()
            }

        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

        if i % 100 == 0:
            sample_text=text[:1]
            token_list=sample_text.masked_select(sample_text != 0).tolist()
            decoded_text=tokenizer.decode(token_list)

            image=dalle.module.generate_images(text[:1], filter_thres=0.9)  # topk sampling at 0.9
            log["image"]=wandb.Image(image, caption=decoded_text)

        if i == 201 and args.flops_profiler:
            raise StopIteration("Profiler has finished running. Stopping training early.")

        wandb.log(log)

    if LR_DECAY: scheduler.step(loss)

    save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

    save_artifact(model_config, DALLE_OUTPUT_FILE_NAME)

save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

wandb.save(DALLE_OUTPUT_FILE_NAME)
save_artifact(model_config, DALLE_OUTPUT_FILE_NAME)
wandb.finish()
