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
It is good practice to work with virtual environments when trying out new code. So please setup a virtual environment using either Python directly or Anaconda, as you our prefer. The code here was developed and tested using [Anaconda](https://docs.anaconda.com/anaconda/) with Python version 3.6. After setting up your environment:

0. Clone this repo:
    ```Shell
    git clone https://github.com/karan-shr/DEXTR-AnnoTool
    cd DEXTR-AnnoTool
    ```
 
1. Install dependencies listed in the reqPackages.txt file:

    If using anaconda then	
    ```Shell
    conda install matplotlib opencv pillow scikit-learn scikit-image h5py
    ```
    For CPU mode:
    ```Shell
    pip install tensorflow keras
    ```
    For GPU mode (CUDA 9.0 and cuDNN 7.0 is required for the latest Tensorflow version. If you have CUDA 8.0 and cuDNN 6.0 installed, force the installation of the vesion 1.4 by using ```tensorflow-gpu==1.4```. More information [here](https://www.tensorflow.org/install/)):
    ```Shell
    pip install tensorflow-gpu keras
    ```
    
  
2. Download the model by running the script inside ```models/```:
    ```Shell
    cd models/
    chmod +x download_dextr_model.sh
    ./download_dextr_model.sh
    cd ..
    ```
    The default model is trained on PASCAL VOC Segmentation train + SBD (10582 images). To download models trained on PASCAL VOC Segmentation train or COCO, please visit our [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr/#downloads), or keep scrolling till the end of this README.

3. To try the demo version of DEXTR, please run:
    ```Shell
    python demo.py
    ```
    If you have multiple GPUs, you can specify which one should be used (for example gpu with id 0):
    ```Shell
    CUDA_VISIBLE_DEVICES=0 python demo.py
    ```

Enjoy!!

### Pre-trained models
We provide the following DEXTR models, pre-trained on:
  * [PASCAL + SBD](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_pascal-sbd.h5), trained on PASCAL VOC Segmentation train + SBD (10582 images). Achieves mIoU of 91.5% on PASCAL VOC Segmentation val.
  * [PASCAL](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_pascal.h5), trained on PASCAL VOC Segmentation train (1464 images). Achieves mIoU of 90.5% on PASCAL VOC Segmentation val.
  * [COCO](https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_coco.h5), trained on COCO train 2014 (82783 images). Achieves mIoU of 87.8% on PASCAL VOC Segmentation val.

### Citation
If you use this code, please consider citing the following papers:

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


We thank the authors of [PSPNet-Keras-tensorflow](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow) for making their Keras re-implementation of PSPNet available!

If you encounter any problems please contact us at {kmaninis, scaelles}@vision.ee.ethz.ch.
