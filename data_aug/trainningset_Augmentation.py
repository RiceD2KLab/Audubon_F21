import os, sys, shutil, glob, csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms
from utils.cropping_train import csv_to_dict, dict_to_csv, crop_img_trainer

#
def flip_img(img, info_dict, output_dir):
    name = ("_flipped.").join(info_dict["file_name"].split("."))

    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
    flipped = transform(img)
    flipped.save(output_dir + "/" + name)

    img_height, img_width, img_depth = info_dict['img_size']

    flipped_dict = {}
    flipped_dict["bbox"] = []
    flipped_dict["file_name"] = name
    flipped_dict["img_size"] = info_dict["img_size"]

    for bbx in info_dict['bbox']:
        instancef_dict = {}
        instancef_dict['class'] = bbx['class']
        instancef_dict['desc'] = bbx['desc']
        instancef_dict['xmin'] = max(img_width - bbx['xmax'], 0)  # Horizontal Flip
        instancef_dict['ymin'] = bbx['ymin']
        instancef_dict['xmax'] = min(img_width - bbx['xmin'], img_width)  # Horizontal Flip
        instancef_dict['ymax'] = bbx['ymax']

        flipped_dict['bbox'].append(instancef_dict)

    dict_to_csv(flipped_dict, empty=False, test = True, output_path=output_dir)



def Test_aug_minor(csv_file,  output_dir, minor_species, overlap, thres, annot_file_ext='csv'):
    file_name = os.path.split(csv_file)[-1][:-4]
    # annotation
    annot_dict = csv_to_dict(csv_path=csv_file, annot_file_ext=annot_file_ext)
    annotation_lst = [list(x.values()) for x in annot_dict['bbox']]

    image_file = csv_file.replace(annot_file_ext, 'JPEG')
    assert os.path.exists(image_file)
    crop_width = 640
    crop_height = 640
    image = Image.open(image_file)
    width, height = image.size

    minors = []
    valid_i = 0
    for dic in annot_dict['bbox']:
        if dic["desc"] in minor_species:
            minors.append(dic)

    for i in range(len(minors)):
        minor = minors[i]
        center_w, center_h = (minor["xmin"] + minor["xmax"]) // 2, (minor["ymin"] + minor["ymax"]) // 2

        left, top, right, bottom = center_w - 0.5 * crop_width, center_h - 0.5 * crop_width, center_w + 0.5 * crop_width, center_h + 0.5 * crop_height
        if left < 0:
            left, right = 0, crop_width
        if right > width:
            left, right = width - crop_width, width
        if top < 0:
            top, bottom = 0, crop_height
        if bottom > height:
            top, bottom = height - crop_height, height


        file_dict = {}
        file_dict["bbox"] = []
        file_dict["img_size"] = (crop_width, crop_height, 3)

        for bbx in annot_dict['bbox']:
            ymin = max(bbx['ymin'] - top, 0)
            ymax = min(bbx['ymax'] - top, crop_height)
            xmin = max(bbx['xmin'] - left, 0)
            xmax = min(bbx['xmax'] - left, crop_width)
            # if the bird is not in this patch, pass
            if xmin > crop_width or xmax < 0 or ymin > crop_height or ymax < 0:  # >=
                continue
            else:
                if (xmax - xmin) * (ymax - ymin) > overlap * (bbx['xmax'] - bbx['xmin']) * (bbx['ymax'] - bbx['ymin']):
                    instance_dict = {}
                    instance_dict['class'] = bbx['class']
                    instance_dict['desc'] = bbx['desc']
                    instance_dict['xmin'] = max(bbx['xmin'] - left, 0)
                    instance_dict['ymin'] = max(bbx['ymin'] - top, 0)
                    instance_dict['xmax'] = min(bbx['xmax'] - left, crop_width)
                    instance_dict['ymax'] = min(bbx['ymax'] - top, crop_height)

                    file_dict['bbox'].append(instance_dict)

        non_minor = 0
        for bbx in file_dict['bbox']:
            if bbx['desc'] not in minor_species:
                non_minor += 1

        if non_minor/len(file_dict['bbox']) > thres:
            continue
        else:
            valid_i += 1
            file_dict["file_name"] = file_name + "_" + str(valid_i).zfill(2) + ".JPEG"
            image.save(output_dir + "/" + file_name + "_" + str(valid_i).zfill(2) + ".JPEG")

            dict_to_csv(file_dict, empty=False, output_path=output_dir, test = True)

            flip_img(img=image, info_dict=file_dict, output_dir=output_dir)



def AugTrainingSet(input_dir, output_dir, minor_species, overlap,
                   thres, annot_file_ext='csv'):
    if annot_file_ext == 'csv':
        Train_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file[-3:] == 'csv']

    # perform both data augmentation and data cropping need for training
    for file in Train_files:#tqdm(Train_files, desc='Cropping files'):
        # data augmentation
        Test_aug_minor(csv_file=file, output_dir=output_dir,
                  minor_species=minor_species, overlap=overlap, thres=thres)