import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_pretrained_od_model(num_classes, choice='fasterrcnn_resnet50_fpn'):
    '''
    Return a pretrained object detection model from torchvision
    Use FastRCNPredictor as box predictor
    '''
    # Choose pretrained object detection model
    if choice == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT',
                                                                     weights_backbone='DEFAULT')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
