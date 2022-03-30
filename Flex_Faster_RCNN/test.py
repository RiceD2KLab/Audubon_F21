import os
import time
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files.faster_rcnn_framework import FasterRCNN, AnchorsGenerator
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from draw_box_utils import draw_box
from backbone.mobile_net_v2 import MobileNetV2
from my_dataset_for_bird import VOCDataSet
from train_utils.coco_utils import get_coco_api_from_dataset
from train_utils.coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils


def create_model(num_classes):
    # resNet50+fpn+faster_RCNN
    # norm_layer should be consistent with training.
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_score_thresh=0.5,
                       image_mean=[0.48119384, 0.46555066, 0.39456555],
                       image_std=[0.17753279, 0.16947103, 0.1736244]
                       )

    # one feature map model
    # backbone = MobileNetV2(norm_layer=torch.nn.BatchNorm2d).features
    # backbone.out_channels = 1280  # num_classes
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    # #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types

# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# create model
model = create_model(num_classes=7)

# load train weights
train_weights = "C://Users//VelocityUser//Documents//Audubon_F21//Flex_Faster_RCNN//save_weights//resNetFpn-model-158.pth"

assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
model.to(device)

# read class_indict
# label_json_path = '/Users/maojietang/Documents/Audubon_F21/Flex_Faster_RCNN/Birds_classes.json'
label_json_path = 'C://Users//VelocityUser//Documents//Audubon_F21//Flex_Faster_RCNN//helper//bird_class.json'
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
json_file = open(label_json_path, 'r')
class_dict = json.load(json_file)
json_file.close()
category_index = {v: k for k, v in class_dict.items()}


cpu_device = torch.device("cpu")
model.eval()
metric_logger = utils.MetricLogger(delimiter="  ")
header = "Test: "

# load validation data set
# VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}
VOC_root = 'C://Users\\VelocityUser\\Documents\\D2K TDS A\\6_class_combine'
val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=12,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=0,
                                                  collate_fn=val_dataset.collate_fn)
data_loader = val_data_set_loader
coco = get_coco_api_from_dataset(data_loader.dataset)
iou_types = _get_iou_types(model)
coco_evaluator = CocoEvaluator(coco, iou_types)

for image, targets in metric_logger.log_every(data_loader, 100, header):
    image = list(img.to(device) for img in image)

