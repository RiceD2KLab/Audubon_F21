''' 
Source for building an object detection model using PyTorch:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 
torchvision.models.detection.faster_rcnn 
'''

import torch
from tqdm import tqdm
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from Audubon_F21.const import COL_NAMES
from Audubon_F21.utils.data_processing import csv_to_df
from Audubon_F21.utils.data_processing import coordinate_to_box


class BirdDataset(torch.utils.data.Dataset):
    ''' Container for bird dataset '''
    def __init__(self, files, transforms=None):
        '''
        Initializes a dataset class instance with image and CSV file names.
        
        Parameters:
            files: A dictionary with keys 'jpg' and 'csv' that contain lists of file paths.
            transforms: A parameter that takes in an image and applies some transformation to it, or None.
        '''
        self.img_files = files['jpg']
        self.csv_files = files['csv']
        self.transforms = transforms

    def __getitem__(self, idx):
        '''
        Retrieves an image and target data corresponding to a given index. 
        
        Parameters:
            idx: Index of the image and target data to retrieve.
        Output: img, target: A tuple containing a Tensor Image and target dictionary for the given index.
            - Tensor Image is a Pytorch tensor with (C, H, W) shape, where C is a number of channels, H and W are image height and width. 
            - The target dictionary contains the following keys:
                    - 'boxes': A PyTorch tensor containing the bounding boxes for each object in the image. The four values represent (x_min, y_min, x_max, y_max)
                               coordinates of the bounding box.
                    - 'labels': A PyTorch tensor containing the class labels for each object in the image. For the bird-only detector, there is only one class.
                    - 'image_id': A PyTorch tensor containing the unique identifier for the image.
                    - 'area': A PyTorch tensor containing the area of each bounding box. 
                    - 'iscrowd': A PyTorch tesor containing booleans for whether each object is a crowd.  
        '''
        # file path
        img_path, csv_path = self.img_files[idx], self.csv_files[idx]
        
        # image
        img = Image.open(img_path).convert("RGB")
        
        # boxes
        box_frame = csv_to_df(csv_path, COL_NAMES)
        num_objs = len(box_frame)
        boxes = []
        for row_idx in range(num_objs):
            x_1 = box_frame.iloc[row_idx]['x']
            y_1 = box_frame.iloc[row_idx]['y']
            width = box_frame.iloc[row_idx]['width']
            height = box_frame.iloc[row_idx]['height']
            boxes.append(coordinate_to_box(x_1, y_1, width, height))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # For bird-only detector, there is only one class 
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # target
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.img_files)
    
def bird_collate_fn(batch):
    ''' 
    Collate function helps PyTorch's DataLoader stack images and targets in batches of consistant size and shape, facilitating more efficient 
    object detection.
    
    Input:
        batch: Batch of Tensor Images is a tensor of (B, C, H, W) shape, where B is a number of images in the batch.  
    Output:
        A tuple of two lists:
            1. The first contains the images in the batch, stacked into a tensor of shape (N, C, H, W), where N is the batch size. 
            2. The second list contains the targets in the batch, where each target is a dictionary.
    '''
    return tuple(zip(*batch))

def get_bird_dataloaders(train_files, test_files):
    '''
    Returns the dataloaders for the train and test datasets.
  
    Input:
        train_files: A dictionary containing paths for the training images and CSV files.
        test_files: A dictionary containing paths for the test images and CSV files.
    
    Output:
        trainloader: A dataloader for the training data.
        testloader: A dataloader for the test data.
    '''
    # Use our dataset and defined transformations
    trainset = BirdDataset(train_files, F.to_tensor)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=bird_collate_fn # Set collate function to our custom function
    ) 

    testset = BirdDataset(test_files, F.to_tensor)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=bird_collate_fn 
    ) 
    return trainloader, testloader

def get_model_and_optim(choice='fasterrcnn_resnet50_fpn'):
    '''
    Input:
        choice: Model choice (we will be using faster R-CNN with a ResNet50 backbone)
    Output:
        model: A pre-trained faster R-CNN model using a ResNet50 backbone network
        optimizer: A stochastic gradient descent (SGD) optimizer with learning rate 0.005, momentum 0.9, and weight decay 0.0005
    '''
    if choice == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    return model, optimizer

def train_model(model, optimizer, trainloader, testloader, n_epochs, device):
    ''' Train a model and print loss for each epoch '''
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch, (images, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1} of {n_epochs}", leave=True, ncols=80)):
            images = list(image.to(device) for image in images)
            targets = [{key: val.to(device) for key, val in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print("Epoch:", epoch + 1, "| Loss:", epoch_loss)
        print()
