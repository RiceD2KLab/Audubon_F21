''' 
Source for building an object detection model using PyTorch:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 
torchvision.models.detection.faster_rcnn 
'''

import torch
from tqdm import tqdm
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .const import COL_NAMES, GROUPS, GROUP_LABELS, SPECIES_LABELS
from .utils.data_processing import csv_to_df, add_col, coordinate_to_box
from .detection import transforms as T
from .detection.coco_eval import CocoEvaluator
from .detection.coco_utils import get_coco_api_from_dataset

def get_transform(train):
    ''' Transformations to apply to images'''
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)

class BirdDataset(torch.utils.data.Dataset):
    ''' Container for bird dataset '''
    def __init__(self, files, choice, transforms=None):
        '''
        Initializes a dataset class instance with image and CSV file names.
        
        Parameters:
            files: A dictionary with keys 'jpg' and 'csv' that contain lists of file paths.
            transforms: A parameter that takes in an image and applies some transformation to it, or None.
        '''
        self.img_files = files['jpg']
        self.csv_files = files['csv']
        self.choice = choice # "bird_only", "group", or "species" 
        self.transforms = transforms

    def __getitem__(self, idx):
        '''
        Retrieves an image and target data corresponding to a given index. 
        
        Parameters:
            idx: Index of the image and target data to retrieve.
        Output: 
            A tuple (img, target)
            - img: Tensor Image is a Pytorch tensor with (C, H, W) shape, where C is a number of channels, H and W are image height and width. 
            - target: Dictionary containing the following keys:
                    - 'boxes': A PyTorch tensor containing the bounding boxes for each object in the image. The four values represent (x_min, y_min, x_max, y_max)
                               coordinates of the bounding box.
                    - 'labels': A PyTorch tensor containing the class labels for each object in the image. For the bird-only detector, there is only one class.
                    - 'image_id': A PyTorch tensor containing the unique identifier for the image.
                    - 'area': A PyTorch tensor containing the area of each bounding box. 
                    - 'iscrowd': A PyTorch tensor containing booleans for whether each object is a crowd.  
        '''
        # file path
        img_path, csv_path = self.img_files[idx], self.csv_files[idx]
        
        # image
        img = Image.open(img_path).convert("RGB")
        box_frame = csv_to_df(csv_path, COL_NAMES)
        num_objs = len(box_frame)
        
        # Add group IDs
        values_dict = {}
        for key, vals in GROUPS.items():
            for val in vals:
                values_dict[val] = key
                
        box_frame = add_col(box_frame, 'group_id', 'class_id', values_dict)
        box_frame = add_col(box_frame, 'group_label', 'group_id', GROUP_LABELS)
        box_frame = add_col(box_frame, 'species_label', 'class_id', SPECIES_LABELS)
        
        # labels
        labels = self.map_label(num_objs, box_frame, self.choice)
        
        # boxes
        boxes = []
        for row_idx in range(num_objs):
            x_1 = box_frame.iloc[row_idx]['x']
            y_1 = box_frame.iloc[row_idx]['y']
            width = box_frame.iloc[row_idx]['width']
            height = box_frame.iloc[row_idx]['height']
            boxes.append(coordinate_to_box(x_1, y_1, width, height))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

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
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.img_files)

    def map_label(self, num_objs, targets_df, choice):
        ''' 
        Maps the class labels to the names of the bird species. 
        '''
        # For bird-only detector, there is only one class 
        if choice == 'bird_only':
            labels = torch.ones((num_objs,), dtype=torch.int64)
        elif choice == 'group':
            labels = torch.tensor(targets_df['group_label'].values, dtype=torch.int64)
        elif choice == 'species':
            labels = torch.tensor(targets_df['species_label'].values, dtype=torch.int64)
        return labels
    
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

def get_bird_dataloaders(train_files, test_files, choice):
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
    trainset = BirdDataset(train_files, choice, get_transform(train=True))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2,
        collate_fn=bird_collate_fn # Set collate function to our custom function
    ) 

    testset = BirdDataset(test_files, choice, get_transform(train=False))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=bird_collate_fn 
    ) 
    return trainloader, testloader

def get_model_and_optim(num_classes, model_choice='fasterrcnn_resnet50_fpn'):
    '''
    Input:
        choice: Model choice (we will be using faster R-CNN with a ResNet50 backbone)
    Output:
        model: A pre-trained faster R-CNN model using a ResNet50 backbone network
        optimizer: A stochastic gradient descent (SGD) optimizer with learning rate 0.005, momentum 0.9, and weight decay 0.0005
    '''
    if model_choice == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    return model, optimizer

def train_model_audubon(model, optimizer, 
                        trainloader, testloader, 
                        n_epochs, device, save_path, model_name):
    ''' Train a model and print loss for each epoch '''
    train_loss_list = []
    test_loss_list = []
    stat_arr = []
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch, (images, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1} of {n_epochs}", leave=True, ncols=80)):
            images = list(image.to(device) for image in images)
            targets = [{key: val.to(device) for key, val in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        epoch_loss = epoch_loss / len(trainloader)

        test_loss = get_test_loss(model, testloader, device)
        train_loss_list.append(epoch_loss)
        test_loss_list.append(test_loss)
        print("Epoch:", epoch + 1, "| Train loss:", epoch_loss, "| Test loss:", test_loss)
        
        # Evaluate model every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            predictions, stats = get_predic_and_eval(model, testloader, device)
            stat_arr.append(stats)
        print()
    
    torch.save(model, save_path + model_name + '.pth')
    return train_loss_list, test_loss_list, predictions, stat_arr

def get_test_loss(model, testloader, device):
    ''' Evaluate a model on the test dataset '''
    test_loss = 0
    with torch.no_grad():
        for batch, (images, targets) in enumerate(testloader):
            images = list(image.to(device) for image in images)
            targets = [{key: val.to(device) for key, val in target.items()} for target in targets]

            loss_dict = model(images, targets)
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            test_loss += losses.item()
    
    return test_loss / len(testloader)
    
def get_predic_and_eval(model, testloader, device):
    ''' Get predictions for the test dataset '''
    model.eval()
    coco = get_coco_api_from_dataset(testloader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    predictions = []
    
    for batch, (images, targets) in enumerate(testloader):
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{key: val.to(device) for key, val in out.items()} for out in outputs]
        predictions += outputs
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
    
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats

    return predictions, stats
