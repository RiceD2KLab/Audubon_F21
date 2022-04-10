from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
import numpy as np
import xml.etree.ElementTree as ET



class VOCDataSet(Dataset):
    """Parse Audubon Birds datasets"""

    def __init__(self, voc_root, year="2022", transforms=None, txt_name: str = "train.txt"):

        # self.root = os.path.join(voc_root, "Test")
        self.root = voc_root
        self.img_root = os.path.join(self.root, 'images')
        self.annotations_root = os.path.join(self.root, 'Annotations')
        # self.img_root = os.path.join(self.root, "JPEGImages")
        # self.annotations_root = os.path.join(self.root, "Annotations_xml")

        # read train.txt or val.txt file
        txt_path = os.path.join('C:\\Users\\VelocityUser\\Documents\\Audubon_F21\\Flex_Faster_RCNN', txt_name)

        # txt_path = 'C://Users//VelocityUser//Documents//Audubon_F21//Flex_Faster_RCNN//helper//train.txt'

        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

        # read class_indict

        json_file = 'C://Users//VelocityUser//Documents//Audubon_F21//Flex_Faster_RCNN//helper//bird_class.json'

        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        json_file.close()

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]

        # print(xml_path)
        # with open(xml_path) as fid:
        #     xml_str = fid.read()
        # xml = etree.fromstring(xml_str)
        tree = ET.parse(xml_path)
        xml = tree.getroot()
        data = self.parse_xml_to_dict(xml)["annotation"]
        # img_path = os.path.join(self.img_root, data["filename"])
        # print(self.img_root)
        img_path = os.path.join(self.img_root, data["filename"].split('\\')[-1])

        image = Image.open(img_path)
        # if image.format != "JPEG":
        #     print(image.format)
        #     raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # Checking the data further, some of the labeled information may have w or h as 0.
            # Such data will lead to the calculation of regression loss as nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]

        # with open(xml_path) as fid:
        #     xml_str = fid.read()
        # xml = etree.fromstring(xml_str)

        tree = ET.parse(xml_path)
        xml = tree.getroot()

        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        Parsing xml files into dictionary form，
        Refer: tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # Iterate through to the bottom and return the information corresponding to the tag directly
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # Recursive traversal of tag information
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # Because there may be more than one object, it needs to be put into a list
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        This method is specially prepared for pycocotools statistics on tags, without any processing of images and tags
        Since there is no need to read the images, the statistics time can be significantly reduced

        Args:
            idx: image's index
        """
        # read xml
        xml_path = self.xml_list[idx]

        # with open(xml_path) as fid:
        #     xml_str = fid.read()
        # xml = etree.fromstring(xml_str)
        tree = ET.parse(xml_path)
        xml = tree.getroot()

        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

# computer train_set mean / std for normalization
def getNormalize(train_data):
    """
    Compute mean and variance for training data
    :param train_data: Dataset(or ImageFolder)
    :return: (mean, std)
    """
    print('Compute mean and variance for training data.')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean_value = torch.zeros(3)
    std_value = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean_value[d] += X[:, d, :, :].mean()
            std_value[d] += X[:, d, :, :].std()
    mean_value.div_(len(train_data))
    std_value.div_(len(train_data))
    print('Computing Complete')
    return list(mean_value.numpy()), list(std_value.numpy())

# Non-experienced bounding box ratios
def K_means_Bird(train_data, K):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)

    # save w and h
    position_info = []

    for _, X in train_loader:
        # w, h
        X = X['boxes'].squeeze()
        for i in range(X.shape[0]):
            position_info.append([(X[i, 2].numpy() - X[i, 0].numpy()), (X[i, 3].numpy() - X[i, 1].numpy())])

    # Initial K-means
    position_info = np.array(position_info)
    K_iter = torch.randperm(len(position_info))[:K]
    centroids = position_info[K_iter, :]

    # initial Label
    Label = np.zeros(position_info.shape[0])
    for i in range(K):
        Label[K_iter[i]] = i + 1

    # Calculate avg_IOU to check converge
    avg_IOU_now = avg_IOU(position_info, centroids)
    avg_IOU_past = 0
    max_iter = 10000
    iteration_k = 0

    # Assign
    while iteration_k <= max_iter and (avg_IOU_now - avg_IOU_past) > 1e-20:
        for i in range(position_info.shape[0]):
            Label[i] = np.argmin(1 - IOU(position_info[i], centroids)) + 1

        # Update centroids
        for j in range(1, K + 1):
            cluster_ind = np.where(j == Label)[0]
            cluster = position_info[cluster_ind, :]
            centroids[j-1, :] = np.mean(cluster, axis=0)
        avg_IOU_past = avg_IOU_now
        avg_IOU_now = avg_IOU(position_info, centroids)
        iteration_k += 1
        # print(iteration_k, (avg_IOU_now - avg_IOU_past))
        if np.isnan(avg_IOU_now):
            assert (np.isnan(avg_IOU_now)), "K = {} is too large for this project.".format(K)
    print('Compute centroids with K = {} for training data, Avg IoU is {}'.format(K, avg_IOU_now))
    return centroids, avg_IOU_now


def IOU(x, centroids):
    """
    :param x: ground truth's w,h
    :param centroids: anchor's w,h set [(w,h),(),...],
    :return: IoU set between ground truth box and all k anchor box
    """
    IoUs = []
    w, h = x  # ground truth's w,h
    for centroid in centroids:
        c_w, c_h = centroid  # anchor's w,h
        if c_w >= w and c_h >= h:  # anchor surrounded by ground truth
            iou = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:  # if anchor is short/wide
            iou = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:  # if anchor is thin and long
            iou = c_w * h / (w * h + c_w * (c_h - h))
        else:  # ground truth surround anchor means both w,h are bigger than c_w and c_h respectively
            iou = (c_w * c_h) / (w * h)

        IoUs.append(iou)  # will become (k,) shape

    return np.array(IoUs)


def avg_IOU(X, centroids):
    """
    :param X: ground truth's w,h set[(w,h),(),...]
    :param centroids: anchor's w,h set[(w,h),(),...]，
    :return: mean value of GT and all anchor (w and h)
    """

    n, d = X.shape
    sum_iou = 0.
    for i in range(X.shape[0]):
        sum_iou += max(IOU(X[i], centroids))

    return sum_iou / n  # get mean

# import transforms
# from draw_box_utils import draw_box
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#
#     json_file = open('C://Users//VelocityUser//Documents//Audubon_F21//Flex_Faster_RCNN//helper//bird_class.json', 'r')
#
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
#
# train_data_set = VOCDataSet('C://Users\\VelocityUser\\Documents\\D2K TDS A\\6_class_combine', "2012", data_transform["train"], "train.txt")
# print(getNormalize(train_data_set))
# print(train_data_set)
#
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()

