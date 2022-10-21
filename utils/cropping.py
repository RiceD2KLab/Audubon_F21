import pandas as pd
from tqdm.auto import tqdm
import cv2
from PIL import Image, ImageDraw
import csv
import os
import shutil
from pathlib import Path
import random
import glob
import numpy as np

def correct_labels(raw_dir):
    """
        Clean the annotation labels: D2K 30 Codes 20221013.xlsx
        
        # MEGRT <--> Replaced by individual species (GREG/REEGWM/SNEG/WHIB/ROSP, etc)
        # MTRNA <--> Replaced by individual tern species (ROTE/SATE/CATE/BGTE/LETE, etc)
        # TCHEA --> TRHEA
    """
    for f in tqdm(glob.glob(raw_dir+'*.bbx')):
        ann = pd.read_csv(f)
        if 'AI Class (Original)' not in ann:
            ann['AI Class (Original)'] = ann['AI Class']
            ann_corr = []
            for lab in ann['AI Class'].values:

                if 0:#np.any([lab.startswith(s) for s in ['GREG','REEGWM','SNEG','WHIB','ROSP']]):
                    # MEGRT <--> Replaced by individual species (GREG/REEGWM/SNEG/WHIB/ROSP, etc)
                    lab_corr = 'MEGRT'
                elif lab == 'ROTE' or lab == 'ROTN':
                    # all royal terns 
                    lab_corr = 'ROT'
                elif lab == 'SANE' or lab == 'SATE':
                    # all sandwich terns 
                    lab_corr = 'SAT'
                elif lab == 'TCHEA':
                    # TCHEA --> TRHEA
                    lab_corr = 'TRHEA'
                elif lab == 'MTRNA':
                    print('mixed tern warning!')
                    lab_corr = lab
                else:
                    lab_corr = lab

                ann_corr.append(lab_corr)

            ann['AI Class'] = np.array(ann_corr)
            ann.to_csv(f, index=False)
        else:
            print('dataset already corrected!')
            break
        

def find_corr_img_file(ann_file):
    """
    Find the image file that has the same file prefix name as teh given file of other formats
    """
    file_ext = ann_file.split('.')[-1]
    im_name_pattern = ann_file.replace('.'+file_ext, '.*')
    im_name = [f for f in glob.glob(im_name_pattern) if f.lower().endswith(('.jpg', '.jpeg'))] # '.png', '.tiff', ...
    assert len(im_name) == 1
    return im_name[0]
    

def csv_to_dict(csv_path, class_map = {}, test=False, annot_file_ext='csv'):
    """
    Function to extract an info dictionary from an xml file
    INPUT:
      csv_path -- path for an csv file, format of bndbox should be xmin, ymin,
                  xmax, ymax
    OUTPUT:
      info_dict -- an info dictionary
    """    
    # load df and rename
    df = pd.read_csv(csv_path, header=0, usecols=range(6), 
                     names=["class_id", "class_name", "x", "y", "width", "height"])
    
    info_dict = {}
    info_dict['bbox'] = []
    info_dict['file_name'] = os.path.split(csv_path)[-1]
    # plotting function needs it, but in JPEG.
    im = cv2.imread(find_corr_img_file(csv_path))

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


def dict_to_csv(info_dict, output_path, empty, img_ext= 'JPEG'):
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
            # if 'Nest' in obj['desc']:
            #     continue
            # if ('Flying' in obj['desc']) or ('Wings Spread' in obj['desc']):
            #     obj['class'] = 'Fly'
            #     obj['desc'] = 'Fly'
            className = obj['class']
            desc = obj['desc']
            xmin = obj['xmin']
            xmax = obj['xmax']
            ymin = obj['ymin']
            ymax = obj['ymax']
            # className, description, xmin, ymin, width, height
            new_bbx_buffer.append([className, desc, int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin)])
    # Name of the file to save
    file_ext = info_dict["file_name"].split('.')[-1]
    assert file_ext.lower() in ['jpeg', 'jpg', 'bbx'], 'Please check file extension!'
    save_file_name = os.path.join(output_path, info_dict["file_name"].replace(file_ext, 'csv'))    

    # print(save_file_name)
    # write to files
    with open(save_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([g for g in schema])
        if not empty:
            writer.writerows(new_bbx_buffer)


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
    subimage_suffix = str(i) + '_' + str(j)
    file_dict[subimage_suffix] = {}
    file_dict[subimage_suffix]['bbox'] = []
    file_dict[subimage_suffix]['file_name'] = info_dict['file_name'][:-4] + '_' + subimage_suffix + '.JPG'
    file_dict[subimage_suffix]['img_size'] = (right - left, bottom - top, 3)

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
                    or (b['xmin'] >= left and b['xmax'] <= right and b['ymin'] >= top and b['ymax'] <= bottom):
                # TODO: The latter condition seems redundant
                valid = True
                # instance_dict is the info_dict for one patch
                instance_dict = {}
                # transform bbx coordinates
                instance_dict['class'] = b['class']
                instance_dict['desc'] = b['desc']
                instance_dict['xmin'] = xmin
                instance_dict['xmax'] = xmax
                instance_dict['ymin'] = ymin
                instance_dict['ymax'] = ymax

                file_dict[subimage_suffix]['bbox'].append(instance_dict)
    return valid


# this function generates all the cropped images and all corresponding label txt files for a single file
# file_dict stores cropped images info dict in one dictionary.
def crop_img(csv_file, crop_height, crop_width, output_dir, class_map = {}, overlap=0.2, annot_file_ext='csv', file_dict={}):
    """
    This function crops one image and output corresponding labels.
    Currently, this function generates the cropped images AND the corresponding csv files to output_dir
    INPUT:
    crop_height, crop_weight -- desired patch size.
    overlap -- threshold for keeping bbx.
    annot_file_ext -- annotation file extension
    """
    info_dict = csv_to_dict(csv_file, class_map, annot_file_ext=annot_file_ext)
    img_height, img_width, _ = info_dict['img_size']
    
    im = Image.open(find_corr_img_file(csv_file), 'r')
    file_name = csv_file.split('/')[-1][:-4]

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
            # however this only saves cropped images that contain birds
            if tile_annot(left, right, top, bottom, info_dict, i, j, crop_height, crop_width, overlap, file_dict):
                c_img = im.crop((left, top, right, bottom))
                c_img.save(os.path.join(output_dir, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j), 'JPEG')
                image = Image.open(os.path.join(output_dir, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j))
                image.save(output_dir + '/' + file_name + '_' + str(i) + '_' + str(j) + '.JPEG')

    # output the file_dict to a folder of csv files containing labels for each cropped file
    for b in file_dict:
        if file_dict[b]['bbox'] == []:
            continue
        else:
            dict_to_csv(file_dict[b], empty=False, output_path=output_dir)

    return file_dict


def crop_dataset(data_dir, output_dir, annot_file_ext = 'csv', class_map = {}, crop_height=640, crop_width=640):
    """
    :param data_dir: image set directory
    :param output_dir: output directory
    :param annot_file_ext: annotation file extension
    :param crop_height: image height after tiling, default 640
    :param crop_width: image width after tiling, default 640
    """

    # intermediate folder for cropped images
    if not os.path.exists(output_dir):
        print(f"Creating output directory at: {output_dir}")
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'Intermediate'))
    elif not os.path.exists(os.path.join(output_dir, 'Intermediate')):
        os.makedirs(os.path.join(output_dir, 'Intermediate'))

    # Load CSV files
    if annot_file_ext == 'csv':
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-3:] == 'csv']
        # TODO: only pass CSV files whose images are in the folder too
    elif annot_file_ext == 'bbx':
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-3:] == 'bbx']
        # TODO: only pass BBX files whose images are in the folder too
    for f in tqdm(files, desc='Cropping files'):
        crop_img(csv_file=f, crop_height=crop_height, crop_width=crop_width, output_dir=output_dir, class_map=class_map,
                 annot_file_ext=annot_file_ext)

    shutil.rmtree(os.path.join(output_dir, 'Intermediate'))


def crop_img_only(img_file_path, output_path, crop_height, crop_width, sliding_size):
    """
    This function crops one image with an adjustable overlap
    INPUT:
    crop_height, crop_weight -- desired patch size.
    """
    # append width, height, depth
    im = cv2.imread(img_file_path)
    img_height, img_width, _ = im.shape
    im = Image.open(img_file_path, 'r')
    _, file_name_full = os.path.split(img_file_path)
    file_name, _ = os.path.splitext(file_name_full)
    # go through the image from top left corner
    for i in range((img_height - crop_height) // sliding_size + 2):

        for j in range((img_width - crop_width) // sliding_size + 2):

            if j < ((img_width - crop_width) // sliding_size + 1) and i < (
                    (img_height - crop_height) // sliding_size + 1):
                left = j * sliding_size
                right = crop_width + j * sliding_size
                top = i * sliding_size
                bottom = crop_height + i * sliding_size

            elif j == ((img_width - crop_width) // sliding_size + 1) and i < (
                    (img_height - crop_height) // sliding_size + 1):
                left = img_width - crop_width
                right = img_width
                top = i * sliding_size
                bottom = crop_height + i * sliding_size

            # if rectangles left on edges, take subimage of crop_height*crop_width by taking a part from within.
            elif i == ((img_height - crop_height) // sliding_size + 1) and j < (
                    (img_width - crop_width) // sliding_size + 1):
                left = j * sliding_size
                right = crop_width + j * sliding_size
                top = img_height - crop_height
                bottom = img_height

            else:
                left = img_width - crop_width
                right = img_width
                top = img_height - crop_height
                bottom = img_height

            c_img = im.crop((left, top, right, bottom))
            c_img.save(os.path.join(output_path, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j), 'JPEG')
            image = Image.open(os.path.join(output_path, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j))
            image.save(os.path.join(output_path, file_name + '_' + str(i) + '_' + str(j) + '.JPEG'))


def crop_dataset_img_only(data_dir, img_ext, output_dir, crop_height=640, crop_width=640, sliding_size=400):
    """
    This function crops image dataset with adjustable sliding size. Bounding boxes are not cropped alongside images.
    Function is to be used during final pipeline stage for prediction
    INPUTS:
        :param data_dir: image set directory
        :param img_ext: image file extension eg. ".JPG"
        :param output_dir: output directory
        :param crop_height: image height after tiling, default 640
        :param crop_width: image width after tiling, default 640
        :param sliding_size: sliding size between each crop, default 400
    """
    
    # intermediate folder for cropped images
    if not os.path.exists(output_dir):
        print(f"Creating output directory at: {output_dir}")
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'Intermediate'))
    elif not os.path.exists(os.path.join(output_dir, 'Intermediate')):
        os.makedirs(os.path.join(output_dir, 'Intermediate'))
    
    # Load CSV files
    files = [d for d in os.listdir(data_dir) if os.path.splitext(d)[1] == img_ext]
    for f in tqdm(files):
        f = os.path.join(data_dir, f)
        crop_img_only(f, output_dir, crop_height, crop_width, sliding_size)
    # remove intermediate folder
    shutil.rmtree(os.path.join(output_dir, 'Intermediate'))


def train_val_test_split(file_dir, output_dir, train_frac=0.8, val_frac=0.1, seed=4):
    """
    :param file_dir: crop_dataset()'s output path:
    :param output_dir: an empty folder
    :param train_frac: fraction for training
    :param val_frac: fraction for validation, 1-train-val will be fraction for test
    """
    
    #making Train, Test, Validate
    phases = ['Train', 'Validate', 'Test']
    [os.makedirs(os.path.join(output_dir,phase), exist_ok=True) for phase in phases]
    
    if not Path(output_dir).is_dir():
        print('The output directory should be an empty folder')
    
    p1, p2, p3 = [os.path.join(output_dir,phase) for phase in phases]
    for phase, p in zip(phases, [p1, p2, p3]):
        if not Path(p).is_dir():
            print('Please create an empty folder named "'+phase+'" inside the output folder')

            
    print(file_dir)
            
    img_list = [f for f in os.listdir(file_dir) if f[-4:] == 'JPEG']
    # random.Random(4).shuffle(img_list)
    random.Random(seed).shuffle(img_list)
    csv_list = [f.replace('JPEG', 'csv') for f in img_list]
    size = len(img_list)
    train_sz = int(size * train_frac)
    val_sz = int(size * val_frac)
    
    print(size, train_sz, val_sz)
    print(len(img_list))
    
    for idx in range(size):
        if idx < train_sz:
            p = p1
        elif idx < train_sz + val_sz:
            p = p2
        else:
            p = p3
        src_img, src_cvs = [os.path.join(file_dir, fn) for fn in (img_list[idx], csv_list[idx])]
        [shutil.move(srcf, p) for srcf in (src_img, src_cvs)]
        

# Added by SP22 to avoid harsh cropping of birds at boundaries
def crop_img_trainer(csv_file, crop_height, crop_width, sliding_size_x, sliding_size_y, output_dir, class_map={}, overlap=0.8, annot_file_ext='csv', file_dict={}, compute_sliding_size=False):
    """
    Function description.
    From other function: 'This function crops one image and output corresponding labels.
    Currently, this function generates the cropped images AND the corresponding csv files to output_dir'
    INPUT:
        crop_height, crop_weight -- desired patch size.
        overlap -- threshold for keeping bbx.
        annot_file_ext -- annotation file extension
    """
    info_dict = csv_to_dict(csv_file, class_map, annot_file_ext=annot_file_ext)
    img_height, img_width, _ = info_dict['img_size']
    im = Image.open(csv_file.replace(annot_file_ext, 'JPG'), 'r')
    file_name = os.path.split(csv_file)[-1][:-4]

    if compute_sliding_size:
        max_w = 0
        max_h = 0
        for b in info_dict['bbox']:
            if b['xmax'] - b['xmin'] > max_w:
                max_w = b['xmax'] - b['xmin']
                print("max_w: ", max_w, "\nxmax: ", b['xmax'], "\nxmin: ", b['xmin'])
            if b['ymax'] - b['ymin'] > max_h:
                max_h = b['ymax'] - b['ymin']
                print("max_h: ", max_h, "\nymax: ", b['ymax'], "\nymin: ", b['ymin'])
        if max_w > 0 and max_h > 0:
            sliding_size_x = crop_width - max_w
            sliding_size_y = crop_height - max_h

    # go through the image from top left corner
    for i in range((img_height - crop_height) // sliding_size_y + 2):
        for j in range((img_width - crop_width) // sliding_size_x + 2):

            if j < ((img_width - crop_width) // sliding_size_x + 1) and i < (
                    (img_height - crop_height) // sliding_size_y + 1):
                left = j * sliding_size_x
                right = crop_width + j * sliding_size_x
                top = i * sliding_size_y
                bottom = crop_height + i * sliding_size_y

            elif j == ((img_width - crop_width) // sliding_size_x + 1) and i < (
                    (img_height - crop_height) // sliding_size_y + 1):
                left = img_width - crop_width
                right = img_width
                top = i * sliding_size_y
                bottom = crop_height + i * sliding_size_y

            # if rectangles left on edges, take subimage of crop_height*crop_width by taking a part from within.
            elif i == ((img_height - crop_height) // sliding_size_y + 1) and j < (
                    (img_width - crop_width) // sliding_size_x + 1):
                left = j * sliding_size_x
                right = crop_width + j * sliding_size_x
                top = img_height - crop_height
                bottom = img_height

            else:
                left = img_width - crop_width
                right = img_width
                top = img_height - crop_height
                bottom = img_height

            # this only saves subimages that have birds
            if tile_annot(left, right, top, bottom, info_dict, i, j, crop_height, crop_width, overlap, file_dict):
                # print('Generating segmentation at position: ', left, top, right, bottom)

                c_img = im.crop((left, top, right, bottom))
                c_img.save(os.path.join(output_dir, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j), 'JPEG')
                image = Image.open(os.path.join(output_dir, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j))
                image.save(output_dir + '/' + file_name + '_' + str(i) + '_' + str(j) + '.JPEG')
                # image.save(os.path.join(output_dir, file_name + '_' + str(i) + '_' + str(j) + '.JPEG'))


    # output the file_dict to a folder of csv files containing labels for each cropped file
    for b in file_dict:
        if file_dict[b]['bbox'] == []:
            continue
        else:
            dict_to_csv(file_dict[b], empty=False, output_path=output_dir)

    return file_dict


def crop_dataset_trainer(data_dir, output_dir, annot_file_ext='csv', class_map={}, crop_height=640, crop_width=640, sliding_size_x=440, sliding_size_y=440, compute_sliding_size=False):
    """
    Function description.
    INPUTS:
        data_dir: image set directory
        output_dir: output directory
        annot_file_ext: annotation file extension
        crop_height: image height after tiling, default 640
        crop_width: image width after tiling, default 640
        sliding_size_x: DESCRIBE, default 440
        sliding_size_y: DESCRIBE, default 440
        compute_sliding_size: If true, computes max sliding size to avoid having cropping birds, default false
    """

    # Intermediate directory for cropped images
    if not os.path.exists(output_dir):
        print(f"Creating output directory at: {output_dir}")
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'Intermediate'))
    elif not os.path.exists(os.path.join(output_dir, 'Intermediate')):
        os.makedirs(os.path.join(output_dir, 'Intermediate'))

    # Load CSV files
    if annot_file_ext == 'csv':
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-3:] == 'csv']
        # TODO: only pass CSV files whose images are in the folder too (update original function too)
    if annot_file_ext == 'bbx':
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-3:] == 'bbx']
        # TODO: only pass BBX files whose images are in the folder too (update original function too)
    for f in tqdm(files, desc='Cropping files'):
        crop_img_trainer(csv_file=f, crop_height=crop_height, crop_width=crop_width, sliding_size_x=sliding_size_x,
                         sliding_size_y=sliding_size_y, output_dir=output_dir, class_map=class_map,
                         annot_file_ext=annot_file_ext, compute_sliding_size=compute_sliding_size)

    # Remove intermediate directory
    shutil.rmtree(os.path.join(output_dir, 'Intermediate'))
