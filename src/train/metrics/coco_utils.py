'''
https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py
'''

from .coco import COCO


def get_coco_eval_api_from_dataset(dataset):
    ''' Returns object detection evaluation api for a given dataset '''
    return convert_to_coco_eval_api(dataset)


def convert_to_coco_eval_api(dataset):
    ''' Converts a dataset to an object detection evaluation api '''
    coco_ds = COCO()

    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1

    # initialize dataset and categories
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()

    # iterate through dataset
    for img_idx in range(len(dataset)):
        img, targets = dataset[img_idx]
        image_id = targets["image_id"].item()

        # add image to dataset
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)

        # get bounding boxes and labels
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]  # convert (x1, y1, x2, y2) to (x1, y1, width, height)
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()

        # add annotations to dataset and update categories
        num_objs = len(bboxes)
        for idx in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[idx]
            ann["category_id"] = labels[idx]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
            categories.add(labels[idx])

    # add categories to dataset
    dataset["categories"] = [{"id": idx} for idx in sorted(categories)]

    # create coco dataset
    coco_ds.dataset = dataset
    coco_ds.create_index()
    return coco_ds
