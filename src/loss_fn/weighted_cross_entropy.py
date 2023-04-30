import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch


def compute_class_weights_from_dataset(dataset):
    ''' Compute class weights from datasets.ImageFolder object '''
    targets = dataset.targets
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(targets),
                                         y=targets)
    return class_weights


def get_weighted_cross_entropy_loss_fn(class_weights, device):
    ''' Return weighted cross entropy loss function '''
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32)).to(device)
    return loss_fn
