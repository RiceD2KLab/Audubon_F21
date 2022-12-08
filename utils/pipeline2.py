from torch.utils.data import Dataset
import pandas as pd
from livelossplot import PlotLosses
import torch
import os
from PIL import Image


def train_test_split(ds_file, train=0.7, test=0.2, valid=0.1, rseed=0):
    '''
    Randomly split the dataset for train and validation based on some ratio
    '''
    dataset = pd.read_csv(ds_file)

    nums = len(dataset)
    train_ = int(train * nums)
    test_ = int(test * nums)
    valid_ = nums - train_ - test_

    train_set = dataset.sample(train_, random_state=rseed).reindex()
    train_set.to_csv(ds_file[:-4]+'-train.csv', index=False) 

    rest = dataset.drop(train_set.index)
    test_set = rest.sample(test_, random_state=rseed+1).reindex()
    test_set.to_csv(ds_file[:-4]+'-test.csv', index=False)

    valid_set = rest.drop(test_set.index).reindex()
    valid_set.to_csv(ds_file[:-4]+'-valid.csv', index=False)


class TernsDataset(Dataset):
    def __init__(self, csv_dir, root_dir, transform=None):
        """
        Args:
        --------
        root_dir (string): Directory with all the images.
        csv_dir (string): The sorce data
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.terns_frame = pd.read_csv(csv_dir, header=0, names=['filename', 'new_class'])
        self.root_dir = root_dir
        self.transform = transform
        self.categories = ['ROT', 'SAT', 'OTH']
        self.categories2ids = {category: id for (id, category) 
                               in enumerate(self.categories)}

    def __len__(self):
        return len(self.terns_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.terns_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        tern_species = self.terns_frame.iloc[idx, 1]
        label_id = self.categories2ids[tern_species]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_id
    
    
def train_model(batch_size, n_epochs, learningRate, model, cost_function, 
                optimizer, scheduler, train_loader, val_loader, save_to, device):

    # Move the model and cost function to GPU (if needed).
    model = model.to(device)
    cost_function = cost_function.to(device)

    # Keep track of best accuracy so far.
    best_accuracy = 0 
    liveloss = PlotLosses()

    # Main for loop of SGD.
    for epoch in range(0, n_epochs):
        logs = {}

        # initialize control variables.
        correct = 0
        cumulative_loss = 0
        n_samples = 0

        # Set the model in training mode.
        model.train()

        # Sample a batch on each iteration.
        for (batch_id, (xb, yb)) in enumerate(train_loader):
            model.zero_grad()

            # Move (x,y) data to GPU (if so desired).
            xb = xb.to(device)
            yb = yb.to(device)

            # Compute predictions.
            predicted = model(xb)

            # Compute loss.
            loss = cost_function(predicted, yb)
            cumulative_loss += loss.item()

            # Count how many correct in batch.
            predicted_ = predicted.detach().softmax(dim = 1)
            max_vals, max_ids = predicted_.max(dim = 1)
            correct += (max_ids == yb).sum().cpu().item()
            n_samples += xb.size(0)

            # Compute gradients (autograd).
            loss.backward()

            # Run one basic training step of SGD.
            optimizer.step()

            # Keep track of loss and accuracy for the plot.
            n_batches = 1 + batch_id 
            logs['loss'] = cumulative_loss / n_batches
            logs['accuracy'] = correct / n_samples

        # initialize control variables.
        correct = 0
        cumulative_loss = 0
        n_samples = 0

        # Set the model in evaluation mode.
        model.eval()

        # No need to keep track of gradients for this part.
        with torch.no_grad():
            # Run the model on the validation set to keep track of accuracy there.
            for (batch_id, (xb, yb)) in enumerate(val_loader):
                # Move data to GPU if needed.
                xb = xb.to(device)
                yb = yb.to(device)

                # Compute predictions.
                predicted = model(xb)

                # Compute loss.
                loss = cost_function(predicted, yb)
                cumulative_loss += loss.item()

                # Count how many correct in batch.
                predicted_ = predicted.detach().softmax(dim = 1)
                max_vals, max_ids = predicted_.max(dim = 1)
                correct += (max_ids == yb).sum().cpu().item()
                n_samples += xb.size(0)

                # Keep track of loss and accuracy for the plot.
                n_batches = 1 + batch_id
                logs['val_loss'] = cumulative_loss / n_batches
                logs['val_accuracy'] = correct / n_samples

        # Save the parameters for the best accuracy on the validation set so far.
        if logs['val_accuracy'] > best_accuracy:
            best_accuracy = logs['val_accuracy']
            torch.save(model.state_dict(), save_to)

        # Update the plot with new logging information.
        liveloss.update(logs)
        liveloss.send()

        # What is this for? Please look it up.
        if scheduler != -1:
            scheduler.step()

    # Load the model parameters for the one that achieved the best val accuracy.
    model.load_state_dict(torch.load(save_to))    
    return model
