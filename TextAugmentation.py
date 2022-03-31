"""Code to create augmented text, and find the distances between them."""
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import spacy
import string

import torch
import torch.nn as nn

from Data import *
from Utils import *

punctuation = list(string.punctuation)

def split_puntuation(s):
    """Returns the word [s] and any of its ending punctuation."""
    return (s[:-1], s[-1]) if s[-1] in punctuation else (s, "")

def token_to_lingkey(t):
    """Returns a linguistically identifying string for token [t]."""
    return f"{t.pos_}{t.tag_}{t.dep_}"

def get_lingkey2words(strs, print_stats=True, memoize=False, use_memoized=False):
    """Returns a dictionary mapping syntatic dependencies to lists of words."""
    if isinstance(use_memoized, str) and os.path.exists(use_memoized):
        tqdm.write(f"Using lingkey2words dictionary loaded from {use_memoized}")
        return torch.load(use_memoized)

    lingkey2words = defaultdict(lambda: [])
    nlp = spacy.load("en_core_web_trf")
    for s in tqdm(strs, desc="Processing captions"):
        doc = nlp(s)
        for t in doc:
            text = split_puntuation(t.text)[0].lower()
            lingkey2words[token_to_lingkey(t)].append(text)

    if print_stats:
        num_keys = len(lingkey2words)
        avg_count = np.mean([len(words) for words in lingkey2words.values()])
        num_with_one = len([w for w in lingkey2words.values() if len(w) == 1])
        tqdm.write(f"Grammatically distinct tokens {num_keys} | Average number of words per token {avg_count} | Number of tokens with one word {num_with_one}")

    if isinstance(memoize, str):
        torch.save(dict(lingkey2words), memoize)
        tqdm.write(f"Saving lingkey2words dictionary to {memoize}")

    return lingkey2words

class BasicTextTransform(nn.Module):
    """Text transform that's not especially linguistically intelligent.
    Theoretically, DALLE can handle semantically (not grammatically) nonsensical
    inputs, though this may be a bigger challenge for BERT.

    ALGORITHM:
    1. Select a random word in a string
    2. Get the linguistic key of the word
    3. Find another word with the linguistic key, and substitute it in place of
        the original word. This is done in a way to match punctuation but not
        (for now) case.
    """
    def __init__(self, captions, apply_n_times=2, use_memoized=False,
        memoize=False, include_original=True):
        super(BasicTextTransform, self).__init__()
        self.include_original = include_original
        self.apply_n_times = apply_n_times
        self.captions = captions
        self.lingkey2words = get_lingkey2words(self.captions,
            use_memoized=use_memoized, memoize=memoize)
        self.nlp = spacy.load("en_core_web_lg")

    def augment_str(self, s):
        """Returns string [s] with an augmentation."""
        word_list = s.split()
        replace_idx = random.randint(0, len(word_list) - 1)
        replaced_word, punc = split_puntuation(word_list[replace_idx])

        doc = self.nlp(s)
        lingkey = token_to_lingkey(doc[replace_idx])
        replacement = random.choice(self.lingkey2words[lingkey])

        word_list[replace_idx] = f"{replacement}{punc}"
        return " ".join(word_list)

    def forward(self, t):
        if self.include_original and self.apply_n_times > 1:
            return [t] + [self.augment_str(t) for _ in range(self.apply_n_times - 1)]
        else:
            return [self.augment_str(t) for _ in range(self.apply_n_times)]

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SPS training")
    P.add_argument("--data_dir", type=str, default=f"{project_dir}/data",
        help="Directory to find data folders int")
    P.add_argument("--data", choices=["coco_captions_images"],
        default="coco_captions_images",
        help="Dataset within --data_dir")
    args = P.parse_args()

    data = TextDataset(f"{args.data_dir}/{args.data}/train", transform="basic",
        include_original=True, apply_n_times=4, return_tokens_too=False)

    for idx in range(10):
        tqdm.write(f"{data[idx]}")
