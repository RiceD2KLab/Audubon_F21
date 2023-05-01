import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50, ResNet50_Weights
import torch


def get_pretrained_od_model(num_classes, choice='fasterrcnn_resnet50_fpn'):
    '''
    Return a pretrained object detection model from torchvision
    Use FastRCNPredictor as box predictor with num_classes output channels

    Args:
        num_classes (int): number of classes in the dataset
        choice (str, optional): name of the pretrained model to use (default: 'fasterrcnn_resnet50_fpn')

    Returns:
        pretrained object detection model
    '''
    # Choose pretrained object detection model
    if choice == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT',
                                                                     weights_backbone='DEFAULT')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_pretrained_resnet50(num_classes, weights=ResNet50_Weights.IMAGENET1K_V2):
    """
    Returns a ResNet50 classifier model from torchvision with num_classes output channels and pretrained weights.

    Args:
        num_classes (int): Number of output channels (i.e. number of classes) for the classifier.
        weights (str): Pretrained weights to be loaded for the ResNet50 model. Default is 'IMAGENET1K_V2'.

    Returns:
        ResNet50 classifier model with specified number of output channels and pretrained weights.
    """
    # Choose pretrained classifier model
    model = resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
