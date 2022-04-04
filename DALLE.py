from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score
import matplotlib.pyplot as plt
import clip
import numpy as np
import torch
import time

def get_generate_images_fn(device, top_k=512, num_candidates=64):
    """Returns a function that returns a list of images from a list of captions.

    Args:
    device          -- the device to generate images on
    top_k           -- top_k parameter
    num_candidates  -- number of candidates to sample from
    """
    dalle = Dalle.from_pretrained('minDALL-E/1.3B').to(device)
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
    clip_model.to(device)

    def generate_images(texts):
        texts = [texts] if isinstance(texts, str) else texts
        result = []
        for t in texts:
            images = dalle.sampling(prompt=t, top_k=top_k,
                num_candidates=num_candidates, device=device).cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            rank = clip_score(prompt=t, images=images, model_clip=clip_model,
                preprocess_clip=preprocess_clip, device=device)
            result.append(torch.from_numpy(images[rank[0]]).permute(2, 0, 1))
        return result

    return generate_images

if __name__ == "__main__":
    f = get_generate_images_fn("cuda:0", top_k=512, num_candidates=64)

    t0 = time.time()
    r1 = f(["This bird has big eyes and black wings"])
    t1 = time.time()
    print("Time to generate one image:", t1 - t0)

    r1 = r1[0]
    r1 = r1.numpy().transpose(1, 2, 0)
    plt.imshow(r1)
    plt.show()
