import pandas as pd
import cv2
from PIL import Image
import csv
import os

def csv_to_dict(csv_path, class_map = {}, annot_file_ext='csv', img_ext = 'jpg'):
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
    
    im = cv2.imread(csv_path.replace(annot_file_ext, img_ext))

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
    if img_ext == 'JPEG':
        save_file_name = os.path.join(output_path, info_dict["file_name"].replace('JPEG', 'csv'))
    if img_ext == 'JPG':
        #img_ext='jpg'
        # print(os.path.join(output_path, info_dict["file_name"]))
        save_file_name = os.path.join(output_path, info_dict["file_name"].replace(img_ext, 'csv'))
    if img_ext == 'bbx':
        save_file_name = os.path.join(output_path, info_dict["file_name"].replace(img_ext, 'csv'))

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
def crop_img(csv_file, img_file, crop_height, crop_width, output_dir, img_ext, class_map = {}, overlap=0.2, annot_file_ext='csv', file_dict={}):
    """
    This function crops one image and output corresponding labels.
    Currently, this function generates the cropped images AND the corresponding csv files to output_dir
    INPUT:
    crop_height, crop_weight -- desired patch size.
    overlap -- threshold for keeping bbx.
    annot_file_ext -- annotation file extension
    """

    info_dict = csv_to_dict(csv_file, class_map, annot_file_ext=annot_file_ext, img_ext = img_ext)
    img_height, img_width, _ = info_dict['img_size']
    im = Image.open(img_file, 'r')
    #print(im)
    # file_name = csv_file.split('/')[-1][:-4]
    file_name = info_dict['file_name'].split('.')[0]
    #print('here', file_name)


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
                #print(os.path.join(output_dir, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j)+ '.jpg')
                c_img.save(os.path.join(output_dir, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j)+ '.jpg')
                image = Image.open(os.path.join(output_dir, 'Intermediate/') + file_name + '_' + str(i) + '_' + str(j)+ '.jpg')
                image.save(output_dir + '/' + file_name + '_' + str(i) + '_' + str(j) + '.jpg')

    # output the file_dict to a folder of csv files containing labels for each cropped file
    for b in file_dict:
        #print(file_dict[b])
        if file_dict[b]['bbox'] == []:
            continue
        else:
            #print('here')
            dict_to_csv(file_dict[b], empty=False, output_path=output_dir, img_ext= 'JPG')


    return file_dict