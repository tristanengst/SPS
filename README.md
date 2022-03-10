# semantic-perceptual-similarity

## Environment Setup
Install the following packages:
```
cython==0.29.25
python=3.9
wandb==0.12.11
```
Then, install the DALL-E package, which installs most of the PyTorch things we need:
```
pip install dalle-pytorch
```

## Data Setup
Download the MS-COCO validation split [images](http://images.cocodataset.org/zips/val2014.zip) and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Unzip both files, and properly format the data:
```
python FormatCOCO.py --annotations PATH_TO_ANNOTATIONS --images PATH_TO_IMAGES
```

## Train DALL-E
We use the implementation here

```
python DALLE/train_dalle.py --taming --image_text_folder data/coco_captions_images --
```
