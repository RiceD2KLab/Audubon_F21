import os, sys, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
from tqdm.autonotebook import tqdm
# from utils.cropping import csv_to_dict # only use this if on PC
from utils.cropping import csv_to_dict
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



def confusion_matrix_report(data_cat_name, predictor, bird_species, img_ext = 'JPEG', iou_thre = 0.5):
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

        annt_bbx = annt_bbx[['x', 'y', 'width', 'height']].to_numpy()
        # compare the anntation and the prediction
        # loop through all ground truth
        for birds in range(annt_bbx.shape[0]):
            # calculating the Iout box
            box = annt_bbx[birds]
            # getting the x2 and y2 values
            box[2] += box[0]
            box[3] += box[1]
            # counter to see if that bird has been found
            ff = 0
            for i in range(len(pred_bbx)):
                if i in found:
                    continue
                # if iou threshold is greater than 50% we have a hit
                iou_val = iou(box, pred_bbx[i])
                if iou_val >= iou_thre:
                    # append the prediction to the list
                    pred_total.append(pred_species[i])
                    found.append(i)
                    ff = 1
                    break
            # if bird is not found then we append an extra category saying it was not found
            truth_total.append(annt[birds]['category_id'])
            if ff == 0:
                pred_total.append(len(bird_species)+1)
                miss_bird += 1

        # plotting the prediction and ground truth
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

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          figure = None):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if figure is None:
        fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi=200)
    else:
        fig, ax = figure
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks, target_names)
        ax.set_yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            ax.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    plt.show()

def getresults(data_cat_name, predictor, bird_species, img_ext = 'JPEG', iou_thre = 0.5):
    '''

    Args:
        data_cat_name: Detectron2's Data Loader
        predictor: the predictor from detectron2. Predicts the outcome based on image
    Returns:
        csv of objects and corresponding predicted class and bounding box
    '''
    data = data_cat_name

    box = 0
    idx = 0

    pred_total = []
    truth_total = []
    pred_bbx_total = []
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

        annt_bbx = annt_bbx[['x', 'y', 'width', 'height']].to_numpy()
        # compare the annotation and the prediction
        # loop through all ground truth
        for birds in range(annt_bbx.shape[0]):
            # calculating the Iout box
            box = annt_bbx[birds]
            # getting the x2 and y2 values
            box[2] += box[0]
            box[3] += box[1]
            # counter to see if that bird has been found
            ff = 0
            for i in range(len(pred_bbx)):
                if i in found:
                    continue
                # if iou threshold is greater than 50% we have a hit
                iou_val = iou(box, pred_bbx[i])
                if iou_val >= iou_thre:
                    # append the prediction to the list
                    pred_total.append(pred_species[i])
                    pred_bbx_total.append(pred_bbx[i])
                    found.append(i)
                    ff = 1
                    break
            # if bird is not found then we append an extra category saying it was not found
            truth_total.append(annt[birds]['category_id'])
            if ff == 0:
                pred_total.append(len(bird_species) + 1)
                miss_bird += 1


    return(pred_total, pred_bbx_total)


