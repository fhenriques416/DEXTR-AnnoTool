# Deep Extreme Cut (DEXTR) based Annotation Tool
is a simple matplotlib-based annotation UI that can be used for extracting segmention masks for images. The main advantage of using this tool is the speed of annotation, as even for complex objects (e.g. the animals in the following image) the segmentation masks can be acquired by annotating only the four extreme points (left-most, right-most, top and bottom) for that object.

<p align="center"><img src="doc/github_teaser.gif" align="center" width=480 height=auto/></p>

This repository builds on the original work of [scaelles/DEXTR-KerasTensorflowPyTorch](https://github.com/scaelles/DEXTR-KerasTensorflow) by providing extra code that converts their demo code into a simple and elegant user-interface for annotating segmentation masks. The tool can be invoked from the command line and offers tracking of annotation progress between subsequent runs i.e. for a given source images folder, it can track which images have already been annotated. The resulting segmentation masks can be saved either as png or as numpy array (.npy) files. 


## Deep Extreme Cut (DEXTR)
The `DEXTR` code used in this fork is a Keras+TensorFlow based reimplementation of original [PyTorch](https://github.com/scaelles/DEXTR-PyTorch) code by the same authors. More information on `DEXTR` can be found on the associated [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr). A short summary is presented as follows: 

![DEXTR](doc/dextr.png)

#### Summary
`DEXTR` is a deep-learning based approach to obtain precise object segmentation in images and videos. To achieve the same, it follows a semi-automatic approach where (usually) user-defined object extreme points are added as an extra channel to an image before it is input into a convolutional neural network (CNN). The CNN learns to transform this information into a segmentation of an object that matches those extreme points. This approach is thus higly useful for guided segmentation (grabcut-style), interactive segmentation, video object segmentation, and dense segmentation annotation.


## Annotation Tool
The annotation tool is a simple and elegant matplotlib-based User-Interface (UI) that allows users to easily and interactively annotate images to obtain object segmentation masks for the same. Some of the main features of this are, as follows:
- easy modification of settings using a YAML-based configuration file.
- ability to annotate extreme points for multiple objects in a image.
- visualization of results for each object during annotation.
- ability to redo annotation if object results are not correct or if wrong points were selected.
- multiple images can be annotated in a session.
- progress tracking across sessions. Thus, only images that were not previously annotated are shown during subsequent runs.


More details on the different settings and using the annotation tool are provided in the later sections. 


## Setup
It is good practice to work with virtual environments when trying out new code. So please setup a virtual environment using either Python directly or Anaconda, as you prefer. The code here was developed and tested using [Anaconda](https://docs.anaconda.com/anaconda/) with Python version 3.6, so we suggest you to do the same. The simplest way to get started is to follow the instructions below.

0. Clone this repo:
    ```Shell
    git clone https://github.com/karan-shr/DEXTR-AnnoTool
    cd DEXTR-AnnoTool
    ```

1. Setting up the environment:
   ```Shell
   conda env create -n dextr_annotool -f conda_env.yml
   ```
   Where `dextr_annotool` is the name of the environment, change it if required, and `conda_env.yml` is the accompanying conda environment file. If you are using pip or installing packages manually, pay attention to the package versions. 
  
2. Download the model by running the script inside ```models/```:
    ```Shell
    cd models/
    chmod +x download_dextr_model.sh
    ./download_dextr_model.sh
    cd ..
    ```
    The default model is trained on PASCAL VOC Segmentation train + SBD (10582 images). To download models trained on PASCAL VOC Segmentation train or COCO, please visit the [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr/#downloads). You can also manually download the models and place them in the models directory.

3. To demo the Annotation UI, please run:
    ```Shell
    python annotate.py -c anno_cfg.yml
    ```
    If you have multiple GPUs, you can specify which one should be used (for example gpu with id 0):
    ```Shell
    CUDA_VISIBLE_DEVICES=0 python annotate.py -c anno_cfg.yml
    ```

4. To starting annotating your dataset, modify the `anno_cfg.yml` configuration file accordingly and run the command mentioned in point 3 and you are good to go... 

## Annotation Tool Configuration

## More information related to DEXTR

### Pre-trained models for DEXTR
We provide the following DEXTR models, pre-trained on:
  * [PASCAL + SBD](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_pascal-sbd.h5), trained on PASCAL VOC Segmentation train + SBD (10582 images). Achieves mIoU of 91.5% on PASCAL VOC Segmentation val.
  * [PASCAL](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_pascal.h5), trained on PASCAL VOC Segmentation train (1464 images). Achieves mIoU of 90.5% on PASCAL VOC Segmentation val.
  * [COCO](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_coco.h5), trained on COCO train 2014 (82783 images). Achieves mIoU of 87.8% on PASCAL VOC Segmentation val.

### Citation
This annotation tool wouldn't have been possible without the excellent work on DEXTR by the team at ETH-Zurich. Please consider citing their following work if you use this tool:

	@Inproceedings{Man+18,
	  Title          = {Deep Extreme Cut: From Extreme Points to Object Segmentation},
	  Author         = {K.K. Maninis and S. Caelles and J. Pont-Tuset and L. {Van Gool}},
	  Booktitle      = {Computer Vision and Pattern Recognition (CVPR)},
	  Year           = {2018}
	}

	@InProceedings{Pap+17,
	  Title          = {Extreme clicking for efficient object annotation},
	  Author         = {D.P. Papadopoulos and J. Uijlings and F. Keller and V. Ferrari},
	  Booktitle      = {ICCV},
	  Year           = {2017}
	}

If you encounter any problems please open an issue.
