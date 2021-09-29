from IPython.display import Image  # for displaying images
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import csv


def csv_to_dict(csv_path, test=False):
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
    info_dict['file_name'] = csv_path.split('/')[-1]

    # plotting function needs it, but in JPEG.
    if test:
        im = cv2.imread(csv_path.replace('csv', 'JPEG'))
    else:
        im = cv2.imread(csv_path.replace('csv', 'JPG'))

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


def dict_to_csv(info_dict, output_path, empty):
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
    save_file_name = os.path.join(output_path, info_dict["file_name"].replace('JPG', 'csv'))

    # write to files
    with open(save_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([g for g in schema])
        if not empty:
            writer.writerows(new_bbx_buffer)

    print(save_file_name)


def plot_img_bbx(image, annotation_lst):
    """
    This is a plotting function to check if the bndbox annotations convertion is done correctly.
    INPUT:
    image -- image path
    annotation_lst -- a list containing all the annotations for this image
    """
    annot = np.array(annotation_lst)
    w, h = image.size
    plotted_img = ImageDraw.Draw(image)

    for annot in annotation_lst:
        obj_class, desc, x_min, y_min, x_max, y_max = annot
        plotted_img.rectangle(((x_min, y_min), (x_max, y_max)), width=3)
        plotted_img.text((x_min, y_min - 10), obj_class)

    plt.figure(figsize=(60, 30))
    plt.imshow(np.array(image), interpolation='nearest')
    plt.show()


def tile_annot(left, right, top, bottom, info_dict, i, j, crop_height, crop_width, overlap):
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


def crop_img(csv_file, crop_height, crop_width, overlap=0.5, output_path='../content/data/crop_test/'):
    """
    This function crops one image and output corresponding labels.
    Currently, this function generates the cropped images AND the corresponding csv files to '../content/data/crop_test'
    INPUT:
    crop_height, crop_weight -- desired patch size.
    overlap -- threshold for keeping bbx.
    """
    info_dict = csv_to_dict(csv_file)
    img_height, img_width, img_depth = info_dict['img_size']
    im = Image.open(csv_file.replace('csv', 'JPG'), 'r')
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
            tile_annot(left, right, top, bottom, info_dict, i, j, crop_height, crop_width, overlap)
            print('Generating segmentation at position: ', left, top, right, bottom)

            c_img = im.crop((left, top, right, bottom))
            c_img.save(os.path.join(output_path, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j), 'JPEG')
            image = Image.open(os.path.join(output_path, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j))
            image.save(output_path + file_name + '_' + str(i) + '_' + str(j) + '.JPEG')

    # output the file_dict to a folder of txt files containing labels for each cropped file
    for b in file_dict:
        if file_dict[b]['bbox'] == []:
            empty = True
        else:
            empty = False
        dict_to_csv(file_dict[b], empty=empty, output_path=output_path)

    return file_dict


if __name__ == 'main':
    file_dict = {}
    file_dict = crop_img('/Users/luominxuan/Desktop/UAV Image - XML - CSV - Ref/DSC06695.csv', crop_height=640,
                         crop_width=640, output_path='/Users/luominxuan/Desktop/UAV Image - XML - CSV - Ref/test')
