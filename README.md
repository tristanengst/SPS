# semantic-perceptual-similarity

## Environment Setup
```
conda create -n py39SPS python=3.9
conda activate py39SPS
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -c conda-forge wandb sentence-transformers tokenizers pyflakes tqdm pytorch-lightning einops omegaconf matplotlib
pip install git+https://github.com/openai/CLIP.git
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```

## Data Setup
**MS-COCO**
Download the MS-COCO validation split [images](http://images.cocodataset.org/zips/val2014.zip) and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). You need to force reload the pages to get the files to actually download. Unzip them, and properly format the data:
```
python FormatCOCO.py --annotations PATH_TO_ANNOTATIONS --images PATH_TO_IMAGES
```
Now, generate text-conditioned images. The following script generates one image for each caption in the dataset. You need to run it many times to generate the augmentations. You should vary the `--start_epoch` input as generated images are named imageN_augSTART_EPOCH; not doing this will cause the script to overwrite what it generates.
```
python GenerateAugmentations --data coco_captions_images --start_epoch 0
```
Each run will take about 1.5 days on an A100 GPU. Be warned. You can modify `generate_images.sh` to do it on a SLURM cluster. Or, you can download the data we used [here](). We didn't use enough data in our experiments, and you should make more. 

**miniImagenet**
Download the data []() here, unzip it, and place it in the `data` folder.

## Method


### Training
We will train on the MS-COCO dataset.

## Experiments
We are interested in the quality of the learned features. Therefore, we evaluate on an out-of-distribution classification task. ie. linear evaluations on miniImagenet. In particular, we are interested in if the semantic supervision helps.
