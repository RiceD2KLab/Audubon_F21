import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50, ResNet50_Weights
import torch


def get_pretrained_od_model(num_classes, choice='fasterrcnn_resnet50_fpn'):
    '''
    Return a pretrained object detection model from torchvision
    Use FastRCNPredictor as box predictor with num_classes output channels
    '''
    # Choose pretrained object detection model
    if choice == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT',
                                                                     weights_backbone='DEFAULT')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_pretrained_resnet50(num_classes, weights=ResNet50_Weights.IMAGENET1K_V2):
    '''
    Return a resnet50 classifier model from torchvision
    with num_classes output channels and pretrained weights
    '''
    # Choose pretrained classifier model
    model = resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
