import pandas as pd
from tqdm.autonotebook import tqdm
import cv2
from PIL import Image, ImageDraw
import csv
import os
import shutil
from pathlib import Path
import random
import boto3
from PIL import Image
import numpy as np


def csv_to_dict_AWS(bucket_name, key,  im_fold, AWS_storage = 's3',annot_file_ext='bbx'):
    """
    Function to extract an info dictionary from an xml file
    INPUT:
      csv_path -- path for an csv file, format of bndbox should be xmin, ymin,
                  xmax, ymax
    OUTPUT:
      info_dict -- an info dictionary
    """
    # all the data should be in this data placement
    s3client = boto3.client(AWS_storage)
    
    response = s3client.get_object(Bucket=bucket_name, Key=key)
    body = response['Body']
    
    df = pd.read_csv(body, header=0, names=["class_id", "class_name", "x", "y", "width", "height"])
    info_dict = {}
    info_dict['bbox'] = []
    
    # get the file name for the annotation
    info_dict['file_name'] = key.split('/')[-1]
    
    imre = s3client.get_object(Bucket = bucket_name, Key= im_fold + info_dict['file_name'].replace(annot_file_ext,'JPG'))
    im = Image.open(imre['Body'])
    im = np.array(im)

    # append width, height, depth
    info_dict['img_size'] = im.shape
    # bndbox info
    for i in range(len(df)):
        # store bbx info for one object
        bbox = {}
        bbox['class'], bbox['desc'], bbox['xmin'], bbox['ymin'], w, h = df.iloc[i,]
        
        bbox['xmax'] = bbox['xmin'] + w
        bbox['ymax'] = bbox['ymin'] + h
        info_dict['bbox'].append(bbox)
        
    return info_dict

def dict_to_csv_AWS(info_dict, output_path, empty):
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
    save_file_name = os.path.join('./temp', info_dict["file_name"].replace('JPG', 'csv'))
    
    
    # write to files
    with open(save_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([g for g in schema])
        if not empty:
            writer.writerows(new_bbx_buffer)
            
    #Write it to the cloud
    s3client.upload_file(Filename = save_file_name, Bucket = my_bucket, Key = output_key +'/'+info_dict["file_name"].replace('JPG', 'csv'))

def tile_annot(left, right, top, bottom, info_dict, i, j, crop_height, crop_width, overlap, file_dict):
    """
    THIS FUNCTION calculate the new positions of bndbox in cropped img and append them to file_dict,
    which is an info dict for that cropped img.

    INPUTS:
    left, right, top, bottom -- params for the python crop img function, coordinates for tiles.
    origin is top left.
    info_dict -- the info_dict we get from the csv_to_dict function.
    overlap -- threshold for keeping a bbox.
    """
    # file_dict stores info of one subimage as a dictionary. keys indicate original file name and subimage position.
    file_dict[str(i) + '_' + str(j)] = {}
    file_dict[str(i) + '_' + str(j)]['bbox'] = []
    file_dict[str(i) + '_' + str(j)]['file_name'] = info_dict['file_name'][:-4] + '_' + str(i) + '_' + str(j) + '.JPG'
    file_dict[str(i) + '_' + str(j)]['img_size'] = (right - left, bottom - top, 3)

    valid = False
    for b in info_dict['bbox']:
        ymin = max(b['ymin'] - top, 0)
        ymax = min(b['ymax'] - top, crop_height)
        xmin = max(b['xmin'] - left, 0)
        xmax = min(b['xmax'] - left, crop_width)
        # if the bird is not in this patch, pass
        if xmin > crop_width or xmax < 0 or ymin > crop_height or ymax < 0:
            continue
        else:
            if (xmax - xmin) * (ymax - ymin) > overlap * (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin']) \
                    or b['xmin'] >= left and b['xmax'] <= right and b['ymin'] >= top and b['ymax'] <= bottom:
                valid = True
                # instance_dict is the info_dict for one patch
                instance_dict = {}
                # transform bbx coordinates
                instance_dict['class'] = b['class']
                instance_dict['desc'] = b['desc']
                instance_dict['xmin'] = max(b['xmin'] - left, 0)
                instance_dict['xmax'] = min(b['xmax'] - left, crop_width)
                instance_dict['ymin'] = max(b['ymin'] - top, 0)
                instance_dict['ymax'] = min(b['ymax'] - top, crop_height)

                file_dict[str(i) + '_' + str(j)]['bbox'].append(instance_dict)
    return valid


# this function generates all the cropped images and all corresponding label txt files for a single file
# file_dict stores cropped images info dict in one dictionary.
def crop_img_AWS(s3client, my_bucket, annot_key, img_key, output_key, crop_height, crop_width, class_map = {}, overlap=0.2, annot_file_ext='csv', file_dict={}):
    """
    This function crops one image and output corresponding labels.
    Currently, this function generates the cropped images AND the corresponding csv files to output_dir
    INPUT:
    crop_height, crop_weight -- desired patch size.
    overlap -- threshold for keeping bbx.
    annot_file_ext -- annotation file extension
    """
    print(img_key)
    print(annot_key)
    info_dict = csv_to_dict_AWS(bucket_name= my_bucket,key = annot_key, im_fold = img_key)
    
    img_height, img_width, img_depth = info_dict['img_size']
    
    image = s3client.get_object(Bucket = my_bucket, Key= img_key + annot_key.split('/')[-1].replace(annot_file_ext, 'JPG') )
    
    im = Image.open(image['Body'], 'r')
    
    file_name = annot_key.split('/')[-1].split('.')[0]
    
    # go through the image from top left corner
    for i in range(img_height // crop_height + 1):

        for j in range(img_width // crop_width + 1):

            if j < (img_width // crop_width) and i < (img_height // crop_height):
                left = j * crop_width
                right = (j + 1) * crop_width
                top = i * crop_height
                bottom = (i + 1) * crop_height

            elif j == img_width // crop_width and i < (img_height // crop_height):
                left = img_width - crop_width
                right = img_width
                top = i * crop_height
                bottom = (i + 1) * crop_height

            # if rectangles left on edges, take subimage of crop_height*crop_width by taking a part from within.
            elif i == img_height // crop_height and j < (img_width // crop_width):
                left = j * crop_width
                right = (j + 1) * crop_width
                top = img_height - crop_height
                bottom = img_height

            else:
                left = img_width - crop_width
                right = img_width
                top = img_height - crop_height
                bottom = img_height
                
            # even if no birds in cropped img, keep the cropped image
            
            if tile_annot(left, right, top, bottom, info_dict, i, j, crop_height, crop_width, overlap, file_dict):
                # print('Generating segmentation at position: ', left, top, right, bottom)
                
                c_img = im.crop((left, top, right, bottom))
                c_img_name = file_name + '_' + str(i) + '_' + str(j) +'.JPEG'
                
                # write all this is the temporary folder in the current working directory
                c_img.save('./temp'+'/'+c_img_name)
                
                #uploading it to the bucket storage
                s3client.upload_file(Filename = './temp'+'/'+c_img_name, Bucket = my_bucket, Key = output_key +'/'+c_img_name)
                
                

    # output the file_dict to a folder of csv files containing labels for each cropped file
    for b in file_dict:
        if file_dict[b]['bbox'] == []:
            empty = True
            continue
        else:
            empty = False
            dict_to_csv_AWS(file_dict[b], empty=empty, output_path=output_key)

    return file_dict


def crop_dataset_AWS(bucket, data_key, output_key, annot_key, annot_file_ext = 'csv', class_map = {}, crop_height=640, crop_width=640):
    """
    :param data_dir: image set directory
    :param output_dir: output directory
    :param annot_file_ext: annotation file extension
    :param crop_height: image height after tiling, default 640
    :param crop_width: image width after tiling, default 640
    """
    s3client = boto3.client('s3') # start grabbing or makign directory in S3 bucket
    
    
    if not key_exist(my_bucket = bucket, my_key = output_key): # this function works only for S3 bucket, will change if we need to modularize this
        print(f"Creating output directory at in S3 bucket called: {output_key}")
        s3client.put_object(Bucket = bucket, Key = output_key)
        
    # making temporary folder in the current directory
    if not os.path.exists('./temp'):
        print(f"Creating temp folder")
        os.makedirs('./temp')

                            

    # find all the files inside the annotated folders                            
    if annot_file_ext == 'csv':
        files = [x['Key'] for x in s3client.list_objects_v2(Bucket = my_bucket, Prefix = annot_key)['Contents']]
    if annot_file_ext == 'bbx':
        files = [x['Key'] for x in s3client.list_objects_v2(Bucket = my_bucket, Prefix = annot_key)['Contents']]
                            
                            
    # for each annotated file, crop the image and place it into the output directory
    for f in tqdm(files, desc='Cropping files'):
        crop_img_AWS(s3client = s3client, my_bucket = bucket, annot_key=f ,img_key = data_key, output_key = output_key,
                     crop_height=crop_height, crop_width=crop_width, class_map=class_map, annot_file_ext=annot_file_ext)
    
    shutil.rmtree('./temp')
    return None

def train_val_test_split_AWS(s3client, my_bucket, file_key, train_key, val_key, test_key, train_frac=0.8, val_frac=0.1):
    """
    :param file_dir: crop_dataset()'s output path:
    :param output_dir: an empty folder
    :param train_frac: fraction for training
    :param val_frac: fraction for validation, 1-train-val will be fraction for test
    """
            
    img_list = [f['Key'] for f in s3client.list_objects(Bucket=my_bucket, Prefix=file_key)['Contents']]
    random.Random(4).shuffle(img_list)
    csv_list = [f.replace('JPEG', 'csv') for f in img_list]
    number_img = len(img_list)
    train_sz = int(number_img * train_frac)
    val_sz = int(number_img * val_frac)
    
   
    s3 = boto3.resource('s3')

    
    for i in range(number_img):
        copy_source_img = {'Bucket': my_bucket, 'Key': img_list[i]}
        copy_source_csv = {'Bucket': my_bucket, 'Key': csv_list[i]}
        if i < train_sz:
            s3.meta.client.copy(copy_source, my_bucket, train_key+img_list[i].split('/')[-1])
            s3.meta.client.copy(copy_source, my_bucket, train_key+csv_list[i].split('/')[-1])
        elif i < train_sz+val_sz:
            s3.meta.client.copy(copy_source, my_bucket, val_key+img_list[i].split('/')[-1])
            s3.meta.client.copy(copy_source, my_bucket, val_key+csv_list[i].split('/')[-1])
        else:
            s3.meta.client.copy(copy_source, my_bucket, test_key+img_list[i].split('/')[-1])
            s3.meta.client.copy(copy_source, my_bucket, test_key+csv_list[i].split('/')[-1])