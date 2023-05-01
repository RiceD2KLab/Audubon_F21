import torch
import os
import numpy as np
from .eval import get_od_loss, get_od_stats, get_clf_loss_accuracy
import sys
from livelossplot import PlotLosses


class HiddenPrints:
    '''
    Hide prints object.
    Example:
        with HiddenPrints():
            print("This will not be printed")
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def train_detector(model, optimizer, loss_fn, n_epochs,
                   trainloader, valloader,
                   device,
                   save_path, name):
    '''
    Trains a detector model for object detection using the specified optimizer, loss function, and training/validation data loaders.

    Args:
        model (torch.nn.Module): The detector model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_fn (torch.nn.Module): The loss function to use for training.
        n_epochs (int): The number of epochs to train for.
        trainloader (torch.utils.data.DataLoader): The data loader for the training set.
        valloader (torch.utils.data.DataLoader): The data loader for the validation set.
        device (torch.device): The device to use for training and inference.
        save_path (str): The path to save the best model.
        name (str): The name of the model.

    Returns:
        Tuple: A tuple of four numpy arrays containing the training loss, validation loss, training statistics, and validation statistics, respectively.

    Raises:
        ValueError: If `save_path` does not exist and cannot be created.

    Notes:
        The model is trained for the specified number of epochs using the given optimizer and loss function. During each epoch, the model is trained on the training data loader by performing forward and backward passes, and then evaluated on the validation data loader by computing the loss and evaluation statistics. The best model is saved based on the validation loss. The function also uses the liveloss library to plot the live training and validation loss.
    '''
    # create save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # initialize variables
    train_loss_list = []
    val_loss_list = []
    train_stats_list = []
    val_stats_list = []
    best_val_loss = float('inf')
    model = model.to(device)

    # plot live loss
    liveloss = PlotLosses()

    for epoch in range(n_epochs):
        logs = {}
        model.train()
        train_loss = 0
        for batch_id, (images, targets) in enumerate(trainloader):
            # move data to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            # backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # evaluate model
        train_loss /= len(trainloader)
        logs['loss'] = train_loss
        val_loss = get_od_loss(model, loss_fn, valloader, device)
        logs['val_loss'] = val_loss

        with HiddenPrints():
            train_stats = get_od_stats(model, trainloader, device)

        with HiddenPrints():
            val_stats = get_od_stats(model, valloader, device)

        # record evaluation metrics
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_stats_list.append(train_stats)
        val_stats_list.append(val_stats)

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, save_path + name + '.pt')

        liveloss.update(logs)
        liveloss.send()

    return train_loss_list, val_loss_list, np.array(train_stats_list), np.array(val_stats_list)


def train_classifier(model, optimizer, loss_fn, n_epochs,
                     trainloader, valloader,
                     device,
                     save_path, name,
                     print_every=5):
    '''
    Trains a PyTorch classifier model and saves the best model based on validation accuracy.

    Args:
        model (torch.nn): The classifier model to train.
        optimizer (torch.optim): The optimizer used for training.
        loss_fn (torch.nn): The loss function used for training.
        n_epochs (int): The number of epochs to train for.
        trainloader (torch.utils.data.DataLoader): The dataloader for the training set.
        valloader (torch.utils.data.DataLoader): The dataloader for the validation set.
        device (torch.device): The device to use for training.
        save_path (str): The path to save the best model.
        name (str): The name of the model to save.
        print_every (int, optional): Print evaluation metrics every `print_every` epochs. Defaults to 5.

    Returns:
        Tuple of four lists representing the training loss, validation loss, training accuracy, and validation accuracy respectively.

    Raises:
        None.
    '''
    # create save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # initialize variables
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    best_val_accuracy = 0

    # Move the model and loss function to device
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    liveloss = PlotLosses()

    for epoch in range(n_epochs):
        logs = {}
        correct = 0
        train_loss = 0
        n_samples = 0

        # Train
        model.train()
        for batch_id, (inputs, labels) in enumerate(trainloader):
            model.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            # Loss
            predicted = model(inputs)
            loss = loss_fn(predicted, labels)
            train_loss += loss.item()

            # Accuracy
            predicted_labels = predicted.detach().softmax(dim=1)
            dummy_max_vals, max_ids = predicted_labels.max(dim=1)
            correct += (max_ids == labels).sum().cpu().item()
            n_samples += inputs.size(0)

            # Backpropagation
            loss.backward()
            optimizer.step()

        train_loss /= len(trainloader)
        train_accuracy = correct / n_samples
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)

        # Evaluate
        val_loss, val_accuracy = get_clf_loss_accuracy(model, loss_fn, valloader, device)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        logs['loss'] = train_loss
        logs['val_loss'] = val_loss
        logs['accuracy'] = train_accuracy
        logs['val_accuracy'] = val_accuracy

        # save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model, save_path + name + '.pt')

        liveloss.update(logs)
        liveloss.send()
    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list
