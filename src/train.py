import torch
from tqdm import tqdm
import os
import numpy as np
from .eval import get_od_loss, get_od_stats


def train_detector(model, optimizer, loss_fn, n_epochs,
                   trainloader, valloader,
                   device,
                   save_path, name,
                   print_every):
    ''' Train a model and save the best model '''
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
                                                          leave=True, ncols=80)):
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
            print("Epoch:", epoch + 1, "| Training loss:", train_loss, "| Validation loss:", val_loss)

        print()
        print("Evaluating model on training set...")
        train_stats = get_od_stats(model, trainloader, device)

        print()
        print("Evaluating model on validation set...")
        val_stats = get_od_stats(model, valloader, device)
        print()

        # record evaluation metrics
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_stats_list.append(train_stats)
        val_stats_list.append(val_stats)

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Updating the best model so far with validation loss:", best_val_loss)
            print()
            torch.save(model, save_path + name + '.pt')

    return train_loss_list, val_loss_list, np.array(train_stats_list), np.array(val_stats_list)
