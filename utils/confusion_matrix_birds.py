import os, sys, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
from tqdm.autonotebook import tqdm
from utils.cropping_hank import csv_to_dict
from PIL import Image, ImageDraw



def iou(box1, box2):
    '''
    this function figures out the bounding box overlap, the iou score
    Args:
        box1: (x1,y1, x2, y2) the bounding box of the ground truth
        box2: (x1,y1, x2, y2) the bounding box of the ground truth

    Returns:
        Iou: the percent of overlap. Higher score is better
    '''

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[-1], box2[-1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (box1[2] - box1[0]) * (box1[-1] - box1[1])
    bb2_area = (box2[2] - box2[0]) * (box2[-1] - box2[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def plot_img_bbx_wrong(image, annotation_lst, annt_pred, img_file):
    w, h = image.size
    plotted_img = ImageDraw.Draw(image)

    # correctly annotated work
    for annot in annotation_lst:
        obj_class, desc, x_min, y_min, x_max, y_max = annot
        plotted_img.rectangle(((x_min, y_min), (x_max, y_max)), width=5, outline=(0, 0, 255))
        # plotted_img.text((x_min, y_min - 10), obj_class, fill = (255,255,255),)

    for annot in annt_pred:
        # print(annot)
        x_min, y_min, x_max, y_max = annot
        # plotted_img.rectangle(((x_min, y_min), (x_max, y_max)), width=5, outline=(255, 0, 0))

    plt.figure(figsize=(20, 10))  # this is for AWS imaging, 60,30 is too big for AWS sagemaker
    # plt.figure(figsize=(60, 30))
    plt.imshow(np.array(image), interpolation='nearest')
    plt.axis('off')
    plt.show()
    # plt.savefig('./wrong/' + img_file)



def confusion_matrix_report(data_cat_name, predictor, bird_species, img_ext = 'JPEG'):
    '''

    Args:
        data_cat_name: Detectron2's Data Loader
        predictor: the predictor from detectron2. Predicts the outcome based on image
    Returns:
        pred_total: An array of predicted bird classes
        truth_total: An array of truth bird classes
    '''

    data = data_cat_name

    box = 0
    idx = 0

    pred_total = []
    truth_total = []
    low_res = []
    nest  =0
    # for each image file, get the image and predict
    for d in tqdm(data):
        # grab the annotation of the files in the data
        annt = d['annotations']
        file = d['file_name']
        im = cv2.imread(file)
        outputs = predictor(im)
        outputs = outputs["instances"].to("cpu")
        lst = list(outputs._fields.items())

        # array of predicted birds (bounding box and species)
        pred_bbx = (lst[0][1].tensor).numpy()
        pred_species = (lst[-1][1]).numpy()

        # print('pred: ', pred_bbx.shape, ' species: ', pred_species.shape)
        # print(pred_bbx.tolist())
        # print('annt: ', annt)

        # keeps track of found birds, if not then it in the not found category
        found = []
        miss_bird = 0
        # get the ground truth
        annt_bbx = pd.read_csv(file.replace(img_ext, 'csv'))

        if 'TRASH' in annt_bbx['class_id'].unique():
            drop_index = []
            for ind, val in enumerate(annt_bbx['class_id']):
                if val == 'TRASH':
                    drop_index.append(ind)

            annt_bbx.drop(index=drop_index, inplace=True)

        # Got rid of all the nest in the cropping section of the model

        # drop_index = []
        # for ind, val in enumerate(annt_bbx['desc']):
        #     if 'Nest' in val:
        #         drop_index.append(ind)
        #         # print(val)
        # annt_bbx.drop(index=drop_index, inplace=True)
        #
        annt_bbx = annt_bbx[['x', 'y', 'width', 'height']].to_numpy()

        # compare the anntation and the prediction
        for birds in range(annt_bbx.shape[0]):
            box = annt_bbx[birds]
            # getting the x2 and y2 values
            box[2] += box[0]
            box[3] += box[1]
            ff = 0
            for i in range(len(pred_bbx)):
                if i in found:
                    continue
                iou_val = iou(box, pred_bbx[i])
                if iou_val >= .5:
                    pred_total.append(pred_species[i])
                    found.append(i)
                    ff = 1
                    break
            truth_total.append(annt[birds]['category_id'])
            if ff == 0:
                pred_total.append(len(bird_species)+1)
                miss_bird += 1

        if annt_bbx.shape[0] > 10:
            if miss_bird / (annt_bbx.shape[0]) == 0:
                w_img = Image.open(file)
                # print(file.replace(img_ext, 'csv'))
                annot_dict = csv_to_dict(csv_path=file.replace(img_ext, 'csv'), test=False, annot_file_ext='csv')
                annt_list = [list(x.values()) for x in annot_dict['bbox']]
                # if ('DJI' not in file.split('/')[-1]) or ('LBN' not in file.split('/')[-1]):
                #     low_res.append(file.split('/')[-1])
                bird_list = [x[1] for x in annt_list]
                # if "Brown Pelican" in bird_list:
                # plot_img_bbx_wrong(w_img, annt_list, annt_pred=pred_bbx.tolist(), img_file=file.split('/')[-1])
    return pred_total, truth_total

