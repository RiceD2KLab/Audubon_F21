import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch


def compute_class_weights_from_dataset(dataset):
    '''
    Computes class weights from a `datasets.ImageFolder` object.
    
    Args:
        dataset (torchvision.datasets.ImageFolder): ImageFolder object containing the dataset
    
    Returns:
        A list of class weights computed using the `compute_class_weight` function
    '''
    targets = dataset.targets

    # Compute class weights for each class in the dataset
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(targets),
                                         y=targets)
    return class_weights


def get_weighted_cross_entropy_loss_fn(class_weights, device):
    ''''
    Returns a weighted cross entropy loss function with the given class weights.
    
    Args:
        class_weights (torch.tensor): A list of class weights
        device (torch.device): The device to use
    
    Returns:
        The weighted cross entropy loss function
    '''
    # Get Cross-entrophy loss with class wieghts
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32)).to(device)
    return loss_fn
