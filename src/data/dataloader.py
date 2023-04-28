import torch
from PIL import Image
from .utils import csv_to_df


class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, jpg_paths, csv_paths, transform, bird_only=True):
        '''
        Args:
            jpg_paths (list of strings): list of paths to images
            csv_paths (list of strings): list of paths to targets
            transforms (torchvision.transforms): transforms to apply to images and targets
        '''
        self._jpg_paths = jpg_paths
        self._csv_paths = csv_paths
        self._transform = transform
        self._bird_only = bird_only

    def __getitem__(self, idx):
        ''' Returns image and target for a given index '''
        # file path
        image_path, target_path = self._jpg_paths[idx], self._csv_paths[idx]

        # image
        image = Image.open(image_path).convert('RGB')

        # labels
        target_df = csv_to_df(target_path)
        if self._bird_only:
            labels = torch.tensor([1] * len(target_df), dtype=torch.int64)
        else:
            labels = torch.tensor(target_df['class_id'].values, dtype=torch.int64)

        # bounding boxes
        boxes = torch.tensor(target_df[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)

        # target
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        # apply transforms
        if self._transform is not None:
            image, target = self._transform(image, target)

        return image, target

    def __len__(self):
        ''' Returns number of images '''
        return len(self._jpg_paths)


def od_collate_fn(batch):
    ''' Stack images and targets in batches of consistant size and shape for object detection '''
    return tuple(zip(*batch))


def get_od_dataloader(jpg_paths, csv_paths, transform, batch_size, shuffle, species):
    ''' Returns dataloader for object detection '''
    od_dataset = ObjectDetectionDataset(jpg_paths, csv_paths, transform, species)
    od_dataloader = torch.utils.data.DataLoader(od_dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                collate_fn=od_collate_fn)
    return od_dataloader