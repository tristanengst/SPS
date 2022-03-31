from PIL import Image
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
import torch
import matplotlib.pyplot as plt

# This notebook supports both CPU and GPU.
# On CPU, generating one sample may take on the order of 20 minutes.
# On a GPU, it should be under a minute.

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda:1')
# Create base glide.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
glide, diffusion = create_model_and_diffusion(**options)
glide.eval()
if has_cuda:
    glide.convert_to_fp16()
glide.to(device)
glide.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in glide.parameters()))
# Create upsampler glide.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
# model_up, diffusion_up = create_model_and_diffusion(**options_up)
# model_up.eval()
# if has_cuda:
#     model_up.convert_to_fp16()
# model_up.to(device)
# model_up.load_state_dict(load_checkpoint('upsample', device))
# print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
def show_images(batch: torch.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))
# Sampling parameters
prompt = "an oil painting of a corgi"
batch_size = 1
guidance_scale = 3.0

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997
##############################
# Sample from the base glide #
##############################

# Create the text tokens to feed to the glide.
tokens = glide.tokenizer.encode(prompt)
tokens, mask = glide.tokenizer.padded_tokens_and_mask(
    tokens, options['text_ctx']
)

# Create the classifier-free guidance tokens (empty)
full_batch_size = batch_size * 2
uncond_tokens, uncond_mask = glide.tokenizer.padded_tokens_and_mask(
    [], options['text_ctx']
)

# Pack the tokens together into glide kwargs.
model_kwargs = dict(
    tokens=torch.tensor(
        [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
    ),
    mask=torch.tensor(
        [mask] * batch_size + [uncond_mask] * batch_size,
        dtype=torch.bool,
        device=device,
    ),
)

# Create a classifier-free guidance sampling function
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = glide(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

# Sample from the base glide.
glide.del_cache()
samples = diffusion.p_sample_loop(
    model_fn,
    (full_batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)[:batch_size]
glide.del_cache()

# # Show the output
# scaled = ((samples + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
# plt.imshow(scaled.squeeze(0).cpu().permute(1, 2, 0)  )
# plt.show()

def glide_generate(prompt):
    glide.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    glide.del_cache()

    return (samples + 1) / 2
