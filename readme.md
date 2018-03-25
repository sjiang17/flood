# Feature Transformation Learning for Occluded Object Detection

Siyu Jiang, Tongzhou Mu, Shixin Li

Object detection under occluded situations is an important yet challenging task, even with the state-of-the-art deep learning methods. Since it is hard to explicitly process various occlusion scenarios in images, we argue that processing with feature maps is more reasonable and efficient. We propose a transformation module to convert feature maps of original images to feature maps corresponding
to an unoccluded pattern. This module can be easily inserted into the feature extraction network of any ConvNetsbased object detection framework. A generative adversarial network structure is adopted in training the proposed module. Experiment results on Pascal VOC and KITTI show that the proposed module is able to improve detection performance on synthetic occlusions. In the future, more investigation can be conducted to improve the performance on real occluded images.

## Preparation 


First of all, clone the code
```
git clone https://github.com/sjiang17/flood.git
```

### prerequisites

* Python 2.7
* Pytorch 0.2.0
* CUDA 8.0 or higher

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
  
More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r pytorch-faster-rcnn/requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd pytorch-faster-rcnn/lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**


### Data Preparation
Download training data on [google drive](https://drive.google.com/open?id=1st_oC-8UngAkS5Y1Q7QD0JRodrW6TVNH)</br>
extract `dataset.tar.gz` and put the `dataset` folder at `./flood/`

Download test data on [google drive](https://drive.google.com/open?id=1j6aY484jJVq9i60vE4jkYsL8kBANEsAz)</br>
extract `VOCdevkit2007.tar.gz` and put the `VOCdevkit2007` folder at `./flood/pytorch-faster-rcnn/data/`

Download trained models on [google drive](https://drive.google.com/open?id=1HEk8A0in5LV3IlKU7WfQy_n4Q341lLW8)</br>
put the `models` folder at `./flood/`


## Train

```
cd flood 
python train_adv.py
```
The models will be saved at `flood/save`

## Test

```
cd flood/pytorch-faster-rcnn
python test_adv.py --dataset pascal_voc --net vgg16 --cuda
```
