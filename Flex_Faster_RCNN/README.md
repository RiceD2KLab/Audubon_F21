# Faster R-CNN for Audubon

## Some modules comes from torchvision source code
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

## Env：
* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools(Linux:```pip install pycocotools```; Windows:```pip install pycocotools-windows```(不需要额外安装vs))
* Ubuntu / Centos
* GPU for Training
* see requirements.txt for more details

## Project Organization：
```
  ├── backbone: extract feature maps (we provide classical backbone and feature pyramid backbone)
  ├── network_files: Faster R-CNN network（including Fast R-CNN module and RPN module and etc）
  ├── train_utils: modules for training and testing（including cocotools）
  ├── my_dataset.py: Customized data sets
  ├── train_mobilenet.py: Use MobileNetV2 as the backbone for training
  ├── train_resnet50_fpn.py: Use resnet50+FPN as the backbone for training
  ├── predict.py: Simple prediction script for prediction testing using trained weights
  ├── validation.py: Validate/test the COCO metrics of the data using the trained weights and generate record_mAP.txt file
  └── pascal_voc_classes.json: pascal_voc tag file
```

## backbones pretrained weights（put them to backbone file）：
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* Or we can use torchvision.models.(vgg16/resnet50, .... with pretrained = True) as the new backbone
 
## training set for fast R-CNN: PASCAL VOC2012 dataset
* Pascal VOC2012 train/val path：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

## Training methods
* Ensure that the data set is prepared in advance
* Ensure that the corresponding pre-trained model weights are downloaded in advance
* To train mobilenetv2+fasterrcnn, use the train_mobilenet.py training script directly
* To train resnet50+fpn+fasterrcnn, use the train_resnet50_fpn.py training script directly

## Notice
* When using the training script, be careful to set '--data-path' (VOC_root) to the **root directory** where you store the 'VOCdevkit' folder
* Since Faster RCNN with FPN structure costs large memory, it is recommended to use the default norm_layer in the create_model function if the GPU memory is not enough (if the batch_size is less than 8).i.e., do not pass the norm_layer variable and go to FrozenBatchNorm2d by default (i.e., do not update the bn layer of the argument)
* When using the prediction script, set 'train_weights' to the path of your own generated weights.
* When using the validation file, be careful to ensure that your validation set or test set must contain targets for each class, and use only the changes '--num-classes', '--data-path' and '--weights'

## Faster RCNN framework
![Faster R-CNN](fasterRCNN.png) 
