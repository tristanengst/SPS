# semantic-perceptual-similarity

## Environment Setup
The order and sources of package installation is nontrivial and important:
1. Create a new conda environment with Python 3.9:
    ```
    conda create -n py39SPS python=3.9
    ```
2. Install PyTorch _through conda_, as well as some other utilities:
    ```
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
    conda install wandb -c conda-forge
    conda install -c conda-forge sentence-transformers
    ```
3. Install the following:
    ```
    tokenizers>=0.10.2
    pyflakes>=2.2.0
    tqdm>=4.46.0
    pytorch-lightning>=1.5
    einops
    omegaconf
    git+https://github.com/openai/CLIP.git
    matplotlib
    ```

## Data Setup
**MS-COCO**
Download the MS-COCO validation split [images](http://images.cocodataset.org/zips/val2014.zip) and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Unzip both files, and properly format the data:
```
python FormatCOCO.py --annotations PATH_TO_ANNOTATIONS --images PATH_TO_IMAGES
```

**miniImagenet**

## Method
As originally formulated, this project involved interesting text augmentation and a corresponding loss function. This relied on a text-to-image model that could produce accurate results for a caption, which published text-to-image models are not yet capable of.

Instead, we take a task from text-and-image data somewhat similar to CLIP, but making full use of text-to-image generation. In CLIP, the basic idea is to jointly encode text and images via contrastive learning; the positive for an image is an encoding of its caption. Here, we take an alternative view: given a caption and two images generated from it, we want to embed the images close together. This allows text-to-image generation to function as a kind of augmentation, standing in contrast to comparatively weak augmentations like random horizontal flips. Within the field of representation learning, this is cool.

However, this removes the linguistic grounding, producing instead a clever way to generate images for contrastive learning. Therefore, we ask how captions might be used as supervision for training. We ought not to use this as an input, as they may not be available at test time. In regular SimCLR-style contrastive learning, images may be pushed apart from images that are in fact their positives as information labeling them as such isn't available—we have nothing more than self-supervision. Here, we have captions. Therefore, instead of pushing negatives apart, we regress them to their captions' distance.

We let the non-grounded baseline be a ablation of this study.

### Training
We will train on the MS-COCO dataset.

## Experiments
We are interested in the quality of the learned features. Therefore, we evaluate on an out-of-distribution classification task. ie. linear evaluations on miniImagenet. In particular, we are interested in if the semantic supervision helps.
