import os, sys, shutil, glob, csv, cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms


def csv_to_dict(csv_path, class_map = {}, test=False, annot_file_ext='csv'):
    """
    Function to extract an info dictionary from an xml file
    INPUT:
      csv_path -- path for an csv file, format of bndbox should be xmin, ymin,
                  xmax, ymax
    OUTPUT:
      info_dict -- an info dictionary
    """
    df = pd.read_csv(csv_path, header=0, names=["class_id", "class_name", "x", "y", "width", "height"])
    info_dict = {}
    info_dict['bbox'] = []
    info_dict['file_name'] = os.path.split(csv_path)[-1]
    # plotting function needs it, but in JPEG.
    if test:
        im = cv2.imread(csv_path.replace('csv', 'JPEG'))
    else:
        im = cv2.imread(csv_path.replace(annot_file_ext, 'JPG'))

    # append width, height, depth
    info_dict['img_size'] = im.shape
    # bndbox info
    for i in range(len(df)):
        # store bbx info for one object
        bbox = {}
        bbox['class'], bbox['desc'], bbox['xmin'], bbox['ymin'], w, h = df.iloc[i,]
        if class_map != {}:
            bbox['class'] = class_map[bbox['desc']]
        bbox['xmax'] = bbox['xmin'] + w
        bbox['ymax'] = bbox['ymin'] + h
        info_dict['bbox'].append(bbox)
    return info_dict


def dict_to_csv(info_dict, output_path, empty, test=False):
    """
    Function to convert (cropped images') info_dicts to annoatation csv files
    INPUT:
     info_dict -- output from the csv_to_dict function, containing bbox, filename, img_size
     output_path -- folder path to store the converted csv files
    OUTPUT:
      an csv file(corresponding for 1 image) saved to a folder. The bndbox info in the format of (className,
      xmin, ymin, width, height)
    """
    new_bbx_buffer = []
    schema = ['class_id', 'desc', 'x', 'y', 'width', 'height']
    if not empty:
        for obj in info_dict['bbox']:
            className = obj['class']
            desc = obj['desc']
            xmin = obj['xmin']
            xmax = obj['xmax']
            ymin = obj['ymin']
            ymax = obj['ymax']
            # className, description, xmin, ymin, width, height
            new_bbx_buffer.append([className, desc, int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin)])
    # Name of the file to save
    if test:
      save_file_name = os.path.join(output_path, info_dict["file_name"].replace('JPEG', 'csv'))
    else:
      save_file_name = os.path.join(output_path, info_dict["file_name"].replace('JPG', 'csv'))
    # write to files
    with open(save_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([g for g in schema])
        if not empty:
            writer.writerows(new_bbx_buffer)
    # print(save_file_name)


def flip_img(img, info_dict, output_dir, command):
    # is_h_flip
    if command[0]:
        # Read info
        img_width, img_height, img_depth = info_dict['img_size']
        name = ("_hflip.").join(info_dict["file_name"].split("."))

        # Flip image
        hflipped = transforms.functional.hflip(img)
        hflipped.save(output_dir+"/"+name)

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
        dict_to_csv(hflip_dict, empty=False, output_path=output_dir, test=True)

    # is_v_flip
    if command[1]:
        # Read info
        img_width, img_height, img_depth = info_dict['img_size']
        name = ("_vflip.").join(info_dict["file_name"].split("."))

        # Flip image
        vflipped = transforms.functional.vflip(img)
        vflipped.save(output_dir+"/"+name)

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
        dict_to_csv(vflip_dict, empty=False, output_path=output_dir, test=True)

    
def rotate_img(img, info_dict, output_dir, command):
    # is_left_rotate
    if command[0]:
        # Read info
        img_width, img_height, img_depth = info_dict['img_size']
        name = ("_90left.").join(info_dict["file_name"].split("."))

        # Rotate image 90 degrees counter clockwise
        lrotated = transforms.functional.rotate(img, 90)
        lrotated.save(output_dir+"/"+name)

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
        dict_to_csv(lrot_dict, empty=False, output_path=output_dir, test=True)
    
    # is_right_rotate
    if command[1]:
        # Read info
        img_width, img_height, img_depth = info_dict['img_size']
        name = ("_90right.").join(info_dict["file_name"].split("."))

        # Rotate image 90 degrees clockwise
        rrotated = transforms.functional.rotate(img, 270)
        rrotated.save(output_dir+"/"+name)

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
        dict_to_csv(rrot_dict, empty=False, output_path=output_dir, test=True)
        
  
def color_img(img, info_dict, output_dir, command):
    if command[0]:
        # Randomly change the brightness and contrast
        jitter = transforms.ColorJitter(brightness=.5, contrast = .3)

        for n in range(1, command[1]+1):
            # Read info
            name = ("_brt"+str(n)+".").join(info_dict["file_name"].split("."))

            # Alter brightness and contrast
            jitted = jitter(img)
            jitted.save(output_dir+"/"+name)

            # Update image name, size and annotation
            jit_dict = info_dict.copy()
            jit_dict["file_name"] = name

            # Save annotation
            dict_to_csv(jit_dict, empty=False, output_path=output_dir, test=True)        
            

def aug_minor(csv_file, crop_height, crop_width, output_dir, minor_species, overlap, thres, aug_command, annot_file_ext='bbx'):
    # Read csv file
    file_name = os.path.split(csv_file)[-1][:-4]

    annot_dict = csv_to_dict(csv_path = csv_file, annot_file_ext=annot_file_ext)
    annotation_lst = [list(x.values()) for x in annot_dict['bbox']]

    # Load image
    image_file = csv_file.replace(annot_file_ext, 'JPG')
    assert os.path.exists(image_file)

    image = Image.open(image_file)
    width, height = image.size

    # Select out all minority birds
    minors = []
    valid_i = 0
    for dic in annot_dict['bbox']:
        if dic["desc"] in minor_species:
            minors.append(dic)

    for i in range(len(minors)):
        minor = minors[i]
        center_w, center_h = (minor["xmin"] + minor["xmax"]) // 2, (minor["ymin"] + minor["ymax"]) // 2

        # Cropping dimensions
        left, top, right, bottom = center_w-0.5*crop_width, center_h-0.5*crop_width, center_w+0.5*crop_width, center_h+0.5*crop_height
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
        file_dict["img_size"] = (crop_width,crop_height,3)  

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
            if bbx['desc'] not in minor_species:
                non_minor += 1

        if non_minor/(len(file_dict['bbox'])) > thres:
            continue
        else:
            valid_i += 1
            file_dict["file_name"] = file_name+"_"+str(valid_i).zfill(2)+ ".JPEG"
            cropped.save(output_dir+"/"+file_name+"_"+str(valid_i).zfill(2)+ ".JPEG")

            dict_to_csv(file_dict, empty=False, output_path=output_dir, test=True)

            # Flipping, rotation and color manipulation
            flip_img(img = cropped, info_dict = file_dict, output_dir = output_dir, command = aug_command[0:2])
            rotate_img(img = cropped, info_dict = file_dict, output_dir = output_dir, command = aug_command[2:4])
            color_img(img = cropped, info_dict = file_dict, output_dir = output_dir, command = aug_command[-1])


def dataset_aug(input_dir, output_dir, minor_species, overlap, thres, aug_command, annot_file_ext = 'bbx', crop_height = 640, crop_width = 640):
    if annot_file_ext == 'bbx':
        files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file[-3:] == 'bbx']
    for file in tqdm(files, desc='Cropping files'):
        aug_minor(csv_file = file, crop_height = crop_height,crop_width = crop_width, output_dir = output_dir,\
                  minor_species = minor_species, overlap = overlap, thres = thres, aug_command = aug_command)
