import torch
from .coco import transforms as T


def get_transform(train):
    '''
    Returns a series of transformations to apply to images, depending on whether
    training is True or False.

    Args:
        train: A boolean value indicating whether the transformations are meant for training
               or testing.

    Returns:
        A torchvision.transforms.Compose object containing the series of transformations to be applied
        to the input image.
    '''
    transforms = []

    # Add transforms of converting PIL to tensor and changing image data type
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))

    # If it's training mode, add some random modifications of images to the transform
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)
