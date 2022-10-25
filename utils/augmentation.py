import os, sys, shutil, glob, csv, cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms
from Audubon_F21.utils.cropping import csv_to_dict, dict_to_csv


def flip_img(img, info_dict, output_dir, command, img_ext):
    """
    Function to flip image and reannotate the coordinates of the birds within
    INPUT:
      img -- image file ready to be flipped
      info_dict -- input image information dictionary
      output_dir -- flipped image and annotation output directory
      command -- flipping command
      img_ext -- file extension specified for saved image
    OUTPUT:
      automatically save processed image and annotation csv files
    """
    # horizontal flipping command
    if command[0]:
        # Read info
        img_width, img_height, img_depth = info_dict['img_size']
        name = ("_hflip.").join(info_dict["file_name"].split("."))

        # Flip image
        hflipped = transforms.functional.hflip(img)
        hflipped.save(output_dir + "/" + name)

        # Update image name, size and annotation
        hflip_dict = {}
        hflip_dict["bbox"] = []
        hflip_dict["file_name"] = name
        hflip_dict["img_size"] = info_dict["img_size"]

        for bbx in info_dict['bbox']:
            instancef_dict = {}
            instancef_dict['class'] = bbx['class']
            instancef_dict['desc'] = bbx['desc']
            instancef_dict['xmin'] = img_width - bbx['xmax']
            instancef_dict['ymin'] = bbx['ymin']
            instancef_dict['xmax'] = img_width - bbx['xmin']
            instancef_dict['ymax'] = bbx['ymax']

            hflip_dict['bbox'].append(instancef_dict)

        # Save Annotation
        dict_to_csv(hflip_dict, output_path = output_dir, empty = False, img_ext = img_ext)
    
    # vertical flipping command
    if command[1]:
        # Read info
        img_width, img_height, img_depth = info_dict['img_size']
        name = ("_vflip.").join(info_dict["file_name"].split("."))

        # Flip image
        vflipped = transforms.functional.vflip(img)
        vflipped.save(output_dir + "/" + name)

        # Update image name, size and annotation
        vflip_dict = {}
        vflip_dict["bbox"] = []
        vflip_dict["file_name"] = name
        vflip_dict["img_size"] = info_dict["img_size"]

        for bbx in info_dict['bbox']:
            instancef_dict = {}
            instancef_dict['class'] = bbx['class']
            instancef_dict['desc'] = bbx['desc']
            instancef_dict['xmin'] = bbx['xmin']
            instancef_dict['ymin'] = img_height - bbx['ymax']
            instancef_dict['xmax'] = bbx['xmax']
            instancef_dict['ymax'] = img_height - bbx['ymin']

            vflip_dict['bbox'].append(instancef_dict)

        # Save Annotation
        dict_to_csv(vflip_dict, output_path=output_dir, empty=False, img_ext=img_ext)


def rotate_img(img, info_dict, output_dir, command, img_ext):
    """
    Function to rotate image and reannotate the coordinates of the birds within
    INPUT:
      img -- image file ready to be rotated
      info_dict -- input image information dictionary
      output_dir -- rotated image and annotation output directory
      command -- rotation command
      img_ext -- file extension specified for saved image
    OUTPUT:
      automatically save processed image and annotation csv files
    """
    # left rotation command
    if command[0]:
        # Read info
        img_width, img_height, img_depth = info_dict['img_size']
        name = ("_90left.").join(info_dict["file_name"].split("."))

        # Rotate image 90 degrees counter clockwise
        lrotated = transforms.functional.rotate(img, 90)
        lrotated.save(output_dir + "/" + name)

        # Update image name, size and annotation
        lrot_dict = {}
        lrot_dict["bbox"] = []
        lrot_dict["file_name"] = name
        lrot_dict["img_size"] = (img_height, img_width, img_depth)

        for bbx in info_dict['bbox']:
            instance_dict = {}
            instance_dict['class'] = bbx['class']
            instance_dict['desc'] = bbx['desc']
            instance_dict['xmin'] = bbx['ymin']
            instance_dict['ymin'] = img_width - bbx['xmax']
            instance_dict['xmax'] = bbx['ymax']
            instance_dict['ymax'] = img_width - bbx['xmin']

            lrot_dict['bbox'].append(instance_dict)

        # Save annotation
        dict_to_csv(lrot_dict, output_path=output_dir, empty=False, img_ext=img_ext)

    # right rotation command
    if command[1]:
        # Read info
        img_width, img_height, img_depth = info_dict['img_size']
        name = ("_90right.").join(info_dict["file_name"].split("."))

        # Rotate image 90 degrees clockwise
        rrotated = transforms.functional.rotate(img, 270)
        rrotated.save(output_dir + "/" + name)

        # Update image name, size and annotation
        rrot_dict = {}
        rrot_dict["bbox"] = []
        rrot_dict["file_name"] = name
        rrot_dict["img_size"] = (img_height, img_width, img_depth)

        for bbx in info_dict['bbox']:
            instance_dict = {}
            instance_dict['class'] = bbx['class']
            instance_dict['desc'] = bbx['desc']
            instance_dict['xmin'] = img_height - bbx['ymax']
            instance_dict['ymin'] = bbx['xmin']
            instance_dict['xmax'] = img_height - bbx['ymin']
            instance_dict['ymax'] = bbx['xmax']

            rrot_dict['bbox'].append(instance_dict)

        # Save annotation
        dict_to_csv(rrot_dict, output_path=output_dir, empty=False, img_ext=img_ext)


def color_img(img, info_dict, output_dir, command, img_ext):
    """
    Function to tune image brightness/contrast and reannotate the coordinates of the birds within
    INPUT:
      img -- image file ready to be tuned
      info_dict -- input image information dictionary
      output_dir -- color-tuned image and annotation output directory
      command -- color tuning command
      img_ext -- file extension specified for saved image
    OUTPUT:
      automatically save processed image and annotation csv files
    """
    # color tuning command
    if command[0]:
        # Randomly change the brightness and contrast
        jitter = transforms.ColorJitter(brightness=.5, contrast=.3)

        for n in range(1, command[1] + 1):
            # Read info
            name = ("_brt" + str(n) + ".").join(info_dict["file_name"].split("."))

            # Alter brightness and contrast
            jitted = jitter(img)
            jitted.save(output_dir + "/" + name)

            # Update image name, size and annotation
            jit_dict = info_dict.copy()
            jit_dict["file_name"] = name

            # Save annotation
            dict_to_csv(jit_dict, output_path=output_dir, empty=False, img_ext=img_ext)


def aug_minor(csv_file, crop_height, crop_width, output_dir, minor_species, overlap, thres, aug_command, img_ext,
              annot_file_ext):
    """
    Function to perform data augmentation on one image file
    INPUT:
      csv_file -- image annotation file
      crop_height/crop_width -- tile size
      output_dir -- augmented images and annotations output directory
      overlap -- minimum portion of a bounding box being accepted in a tile
      thres -- threshold of non-minor creatures existing in a tile
      aug_command -- augmentation methods
      minor_species -- species that we want to augment
      img_ext -- file extension specified for saved image
      annot_file_ext -- annotation file extension
    OUTPUT:
      automatically save cropped tiles and annotation files
    """
    # Read csv file
    file_name = os.path.split(csv_file)[-1][:-4]

    annot_dict = csv_to_dict(csv_path=csv_file, annot_file_ext=annot_file_ext, img_ext = img_ext)
    annotation_lst = [list(x.values()) for x in annot_dict['bbox']]

    # Load image
    image_file = csv_file.replace(annot_file_ext, img_ext)
    assert os.path.exists(image_file)

    image = Image.open(image_file)
    width, height = image.size

    # Select out all minority birds
    minors = []
    valid_i = 0
    for dic in annot_dict['bbox']:
        if dic["class"] in minor_species:
            minors.append(dic)
    # print(minors)
    for i in range(len(minors)):
        minor = minors[i]
        center_w, center_h = (minor["xmin"] + minor["xmax"]) // 2, (minor["ymin"] + minor["ymax"]) // 2

        # Cropping dimensions
        left, top, right, bottom = center_w - 0.5 * crop_width, center_h - 0.5 * crop_width, center_w + 0.5 * crop_width, center_h + 0.5 * crop_height
        if left < 0:
            left, right = 0, crop_width
        if right > width:
            left, right = width - crop_width, width
        if top < 0:
            top, bottom = 0, crop_height
        if bottom > height:
            top, bottom = height - crop_height, height

        # Crop the minority bird
        cropped = image.crop((left, top, right, bottom))

        # Image name, size and annotation
        file_dict = {}
        file_dict["bbox"] = []
        file_dict["img_size"] = (crop_width, crop_height, 3)

        for bbx in annot_dict['bbox']:
            ymin = max(bbx['ymin'] - top, 0)
            ymax = min(bbx['ymax'] - top, crop_height)
            xmin = max(bbx['xmin'] - left, 0)
            xmax = min(bbx['xmax'] - left, crop_width)

            if xmin > crop_width or xmax < 0 or ymin > crop_height or ymax < 0:
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

        # Set majority birds threshold
        non_minor = 0
        for bbx in file_dict['bbox']:
            if bbx['class'] not in minor_species:
                non_minor += 1

        if non_minor / (len(file_dict['bbox'])) > thres:
            continue
        else:
            valid_i += 1
            file_dict["file_name"] = file_name + "_" + str(valid_i).zfill(2) + "."+img_ext
            # cropped.save(output_dir + "/" + file_name + "_" + str(valid_i).zfill(2) + "."+img_ext)
            # # print(file_dict["file_name"] )
            # dict_to_csv(file_dict, output_path=output_dir, empty=False, img_ext=img_ext)

            # Flipping, rotation and color manipulation
            flip_img(img=cropped, info_dict=file_dict, output_dir=output_dir, command=aug_command[0:2], img_ext= img_ext)
            rotate_img(img=cropped, info_dict=file_dict, output_dir=output_dir, command=aug_command[2:4], img_ext= img_ext)
            color_img(img=cropped, info_dict=file_dict, output_dir=output_dir, command=aug_command[-1], img_ext= img_ext)


def dataset_aug(input_dir, output_dir, minor_species, overlap, thres, aug_command, img_ext, annot_file_ext,
                crop_height=640, crop_width=640):
    """
    Function to perform data augmentation on a dataset
    INPUT:
      input_dir -- input directory of images and annotation files
      output_dir -- output directory of augmented images and annotation files
    OUTPUT:
      automatically save cropped tiles and annotation files
    """
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file[-3:] == annot_file_ext]
    for file in tqdm(files, desc='Aug_files'):
        aug_minor(csv_file=file, crop_height=crop_height, crop_width=crop_width, output_dir=output_dir,
                  minor_species=minor_species, overlap=overlap, thres=thres, annot_file_ext=annot_file_ext,
                  img_ext=img_ext, aug_command=aug_command)
