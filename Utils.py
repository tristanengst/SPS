import os
import string
from tqdm import tqdm
import torch

project_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

num_workers = 6

# Turn off WandB console logging, since we don't need it and it breaks TQDM.
os.environ["WANDB_CONSOLE"] = "off"

# Make non-determinism work out. This function should be called first
def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tqdm.write(f"Set the NumPy, PyTorch, and Random modules seeds to {seed}")
    elif isinstance(seed, dict):
        random.setstate(seed["random_seed"])
        np.random.set_state(seed["numpy_seed"])
        torch.set_rng_state(seed["pytorch_seed"])
        tqdm.write(f"Reseeded program with old seed")
    else:
        raise ValueError(f"Seed should be int or contain resuming keys")

def preprocess_str(s):
    return s.translate(str.maketrans("", "", string.punctuation)).lower()
