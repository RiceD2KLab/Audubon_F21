import torch


def get_sgd_optim(model, lr, momentum=0.9, weight_decay=0.0005):
    '''
    Return a SGD optimizer associated with the input model with learning rate lr,
    momentum momentum and weight decay weight_decay
    '''
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer
