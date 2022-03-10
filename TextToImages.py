import argparse
import torch
from dalle_pytorch import OpenAIDiscreteVAE, DALLE, VQGanVAE

vae = OpenAIDiscreteVAE()       # loads pretrained OpenAI VAE

dalle = DALLE(
    dim = 1024,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = 10000,    # vocab size for text
    text_seq_len = 256,         # text sequence length
    depth = 1,                  # should aim to be 64
    heads = 16,                 # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
)

text = torch.randint(0, 10000, (4, 256))
images = torch.randn(4, 3, 256, 256)

loss = dalle(text, images, return_loss = True)
loss.backward()

def one_epoch(model, optimizer, loader):
    """
    """

if __name__ == "__main__":
    P = argparse.ArgumentParser("")
    P.add_argument("--data_dir", type=str, default=data_dir,
        help="Directory for storing project data. Only override if nondefault")
    P.add_argument("--data", default=["coco_captions_images"],
        help="Path to folder containing training images and captions")
    P.add_argument("--seed", type=int, default=0,
        help="Random seed for dataset splitting")

    P.add_argument("--bs", type=int, default=1,
        help="Batch size")
    P.add_argument("--lr", type=float, default=4.5e-4,
        help="Learning rate")
    P.add_argument("--vae", choices=["vqgan", "openai"], default="vqgan",
        help="Pretrained VAE type")
    P.add_argument("--options", nargs="+", default=[],
        help="Options")
    args = P.parse_args()

    args.options = sorted([

    ])

    set_seed(args.seed)
    run_id = wandb.util.generate_id()
    wandb.init(anonymous="allow", id=run_id, project="DALL-E Training",
        mode="online" if args.wandb else "disabled", config=args,
        name=save_dir.replace(f"{project_dir}/generators/", ""))
