# FeatEnHancer for low-light object detection
## Usage and Installation 

Our Implementation is based on [Detectron2](https://github.com/facebookresearch/detectron2), please refer to [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for more details.

Install the Detectron2:

```
git clone https://github.com/facebookresearch/detectron2.git

python setup.py build develop
```


## Data Preparation

Create a new folder named "exdark" in the  ```low-light-object-detection-detectron2/data``` folder.

Create a new folder named "darkface" in the ```low-light-object-detection-detectron2/data``` folder.


Download the [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) dataset and copy the images into ```low-light-object-detection-detectron2/data/exdark/images/```.

Download the [DARK FACE](https://flyywh.github.io/CVPRW2019LowLight/) dataset and copy the images into ```low-light-object-detection-detectron2/data/darkface/images/```.

## Evaluation

```
sh low-light-object-detection-detectron2/test_exdark.sh
sh low-light-object-detection-detectron2/test_darkface.sh
```


## Training

```
sh low-light-object-detection-mmdetection/exec_script_exdark.sh
sh low-light-object-detection-mmdetection/exec_script_darkface.sh
```

## Acknowledgements

Our implementation is based on [detectron2](https://github.com/facebookresearch/detectron2) and [Featurized Query R-CNN](https://github.com/hustvl/Featurized-QueryRCNN), we thank for their open-source code.





