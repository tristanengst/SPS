import os
from torch.utils.data import Dataset

from TextAugmentation import *
from Tokenizer import SimpleTokenizer
from Utils import *

def read_text_files_to_list(folder):
    """Returns a list where the [ith] element is the string that is the [ith]
    text file in [folder] according to alphabetization.
    """
    text_files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    result = []
    for file in text_files:
        with open(f"{folder}/{file}", "r") as f:
            lines = f.read().split("/n")
            result += lines

    return result

class TextDataset(Dataset):
    """Dataset for returning strings of text. To avoid GPU difficulties, finding
    the distance between returned strings must be handled somewhere else.

    Args:
    source          -- folder containing text and image files
    transform       -- function for augmenting strings of text, or string
                        describing it, eg. 'basic'
    """
    def __init__(self, source, transform="basic", apply_n_times=2, memoize=True,
            use_memoized=True, include_original=True, return_tokens_too=True,
            **kwargs):
        super(TextDataset, self).__init__()
        self.tokenizer = SimpleTokenizer()
        self.return_tokens_too = return_tokens_too
        self.data = read_text_files_to_list(source)

        if transform == "basic":
            self.transform = BasicTextTransform(self.data,
                apply_n_times=apply_n_times,
                include_original=include_original,
                use_memoized=f"{source}/lingkey2words.pt" if use_memoized else False,
                memoize=f"{source}/lingkey2words.pt" if memoize else False)
        else:
            self.transform = transform

    def __len__(self): return len(self.data)

    def tokenize(self, s): return self.tokenizer.tokenize(s, 256).squeeze(0)

    def __getitem__(self, idx):
        if self.return_tokens_too:
            results = self.transform(self.data[idx])
            return results, [self.tokenize(r) for r in results]
        else:
            return self.transform(self.data[idx])
