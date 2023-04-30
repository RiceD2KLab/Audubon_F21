import torch
from tqdm import tqdm
import os
import numpy as np
from .eval import get_od_loss, get_od_stats, get_clf_loss_accuracy
import sys


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
                   save_path, name,
                   print_every=5):
    '''
    Args:
        model (torch.nn): detector model
        optimizer (torch.optim): optimizer
        loss_fn (torch.nn): loss function
        n_epochs (int): number of epochs
        trainloader (torch.utils.data.DataLoader): dataloader for training set
        valloader (torch.utils.data.DataLoader): dataloader for validation set
        device (torch.device): device to use
        save_path (string): path to save the best model
        name (string): name of the model
        print_every (int): print evaluation metrics every print_every epochs
    Train a model and save the best model.
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

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_id, (images, targets) in enumerate(tqdm(trainloader,
                                                          desc=f"Epoch {epoch + 1} of {n_epochs}",
                                                          position=0, leave=True, ncols=80)):
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
        val_loss = get_od_loss(model, loss_fn, valloader, device)

        # print evaluation metrics
        if (epoch + 1) % print_every == 0 or epoch == n_epochs - 1:
            print()
            print("Epoch:", epoch + 1, "| Training loss:", f'{train_loss:.4f}', "| Validation loss:", f'{val_loss:.4f}')

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
            print()
            print("Updating the best model so far with validation loss:", best_val_loss)
            torch.save(model, save_path + name + '.pt')

    return train_loss_list, val_loss_list, np.array(train_stats_list), np.array(val_stats_list)


def train_classifier(model, optimizer, loss_fn, n_epochs,
                     trainloader, valloader,
                     device,
                     save_path, name,
                     print_every=5):
    '''
    Args:
        model (torch.nn): classifier model
        optimizer (torch.optim): optimizer
        loss_fn (torch.nn): loss function
        n_epochs (int): number of epochs
        trainloader (torch.utils.data.DataLoader): dataloader for training set
        valloader (torch.utils.data.DataLoader): dataloader for validation set
        device (torch.device): device to use
        save_path (string): path to save the best model
        name (string): name of the model
        print_every (int): print evaluation metrics every print_every epochs
    Train a model and save the best model
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

    for epoch in range(n_epochs):
        correct = 0
        train_loss = 0
        n_samples = 0

        # Train
        model.train()
        for batch_id, (inputs, labels) in enumerate(tqdm(trainloader,
                                                         desc=f"Epoch {epoch + 1} of {n_epochs}",
                                                         position=0, leave=True, ncols=80)):
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

        # print evaluation metrics
        if (epoch + 1) % print_every == 0 or epoch == n_epochs - 1:
            print()
            print("Epoch:", epoch + 1, "| Training loss:", f'{train_loss:.4f}', "| Validation loss:", f'{val_loss:.4f}',
                  "| Training accuracy:", f'{train_accuracy:.4f}', "| Validation accuracy:", f'{val_accuracy:.4f}')

        # save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print()
            print("Updating the best model so far with validation accuracy:", best_val_accuracy)
            torch.save(model, save_path + name + '.pt')
    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list
