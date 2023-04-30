import torch


def get_adam_optim(model, lr=0.0001):
    ''' Return an ADAM optimizer associated with the input model with learning rate lr'''
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    return optimizer
