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

def collate_fn(list_of_inputs):
    return [x[0] for x in list_of_inputs] + [x[1] for x in list_of_inputs]

class HumanAugmentationTextDataset(Dataset):
    """
    """

    def __init__(self, source, num_samples=2, data_path=f"{project_dir}/data"):
        super(HumanAugmentationTextDataset, self).__init__()
        self.num_samples = num_samples
        self.file2texts = []
        self.files = sorted([f for f in os.listdir(f"{data_path}/{source}") if f.endswith(".txt")])
        for file in self.files:
            with open(f"{data_path}/{source}/{file}", "r") as f:
                texts = f.read().split("\n")
                self.file2texts.append([t for t in texts if not t == ""])

    def __len__(self): return len(self.file2texts)

    def __getitem__(self, idx):
        return random.sample(self.file2texts[idx], self.num_samples)

class AugmentedTextDataset(Dataset):
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
        super(AugmentedTextDataset, self).__init__()
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
