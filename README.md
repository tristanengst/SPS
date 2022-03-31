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
    conda install -c conda-forge spacy
    conda install -c conda-forge cupy
    python -m spacy download en_core_web_lg
    conda install -c conda-forge sentence-transformers
    ```
3. Install GLIDE (filtered).

## Data Setup
Download the MS-COCO validation split [images](http://images.cocodataset.org/zips/val2014.zip) and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Unzip both files, and properly format the data:
```
python FormatCOCO.py --annotations PATH_TO_ANNOTATIONS --images PATH_TO_IMAGES
```






## Method
As originally formulated, this project involved interesting text augmentation and a corresponding loss function. This all relied on a good text-to-image model, which apparently doesn't exist. I've decided that the best option is GLIDE (filtered), but this doesn't do an adequate job of generating images that actually match an input caption to make the proposed method work.

**There are a few different things to be done:**
1. Large batch training isn't possible; GLIDE is too slow. We can just make batch sizes tiny (eg. 8) and hope for the best.
2. To increase image quality, we can drop text augmentation. This allows precomputing text distances. 
3. A worthwhile direction is something that amounts to roughly CLIP as a _perceptual_ distance metric. This could literally be CLIP + LPIPS. 