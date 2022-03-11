# semantic-perceptual-similarity

## Environment Setup
The order and sources of package installation is nontrivial and important:
1. Create a new conda environment with Python 3.9:
    ```
    conda create -n py39SPS python=3.9
    ```
2. Install PyTorch _through conda_, as well as some other utiltiies:
    ```
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
    conda install wandb -c conda-forge
    conda install tqdm cython
    ```
3. Install `dalle-pytorch`:
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
