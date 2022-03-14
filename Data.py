import os
from torch.utils.data import Dataset

from TextAugmentation import *
from Utils import *

datasets = []

def read_text_files_to_list(folder):
    """Returns a list where the [ith] element is the string that is the [ith]
    text file in [folder] according to alphabetization.
    """
    text_files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    result = [None] * len(text_files)
    for idx,file in enumerate(text_files):
        with open(f"{folder}/{file}", "r") as f:
            result[idx] = f.read()

    return result

class TextDataset(Dataset):
    """Dataset for returning strings of text. To avoid GPU difficulties, finding
    the distance between returned strings must be handled somewhere else.
BasicTextTransform
    Args:
    source      -- folder containing text and image files
    transform   -- function for augmenting strings of text. Should return
                    multiple augmentations if this is needed
    """

    def __init__(self, source, transform="basic", apply_n_times=2, memoize=True,
            use_memoized=True, include_original=True):
        super(TextDataset, self).__init__()

        self.data = read_text_files_to_list(source)
        if transform == "basic":
            self.transform = BasicTextTransform(self.data,
                apply_n_times=apply_n_times,
                use_memoized=f"{source}/lingkey2words.pt" if use_memoized else False,
                memoize=f"{source}/lingkey2words.pt" if memoize else False,
                include_original=include_original)
        else:
            self.transform = transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.transform(self.data[idx])
