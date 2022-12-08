import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm.auto import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def confusion_matrix_report(data_cat_name, predictor, bird_species, img_ext = 'JPEG', iou_thre = 0.5, save_to = None):
    '''
    Args:
    --------
    data_cat_name: Detectron2's Data Loader
    predictor: the predictor from detectron2. Predicts the outcome based on image
        
    Returns:
    --------
    pred_total : An array of predicted bird classes
    truth_total: An array of truth bird classes
    '''
    data = data_cat_name
    lab_cls2cat = dict(zip(bird_species, range(len(bird_species))))

    pred_total = []
    truth_total = []
    tot_cnt  =0
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
                
        # get the ground truth
        annt_df = pd.read_csv(file.replace(img_ext, 'csv'))

        change_index = []
        for ind, val in enumerate(annt_df['class_id']):
            if val not in bird_species[:-1]:
                change_index.append(ind)
        annt_df.loc[change_index, 'class_id'] = bird_species[-1]
        
        annt_lab = annt_df['class_id'].to_numpy()
        annt_bbx = annt_df[['x', 'y', 'width', 'height']].to_numpy()
        
        # compare the anntation and the prediction loop through all ground truth
        for birds in range(annt_bbx.shape[0]):
            box = annt_bbx[birds]
            # getting the x2 and y2 values
            box[2] += box[0]
            box[3] += box[1]
            
            # counter to see if that bird has been found
            pred_voter = []
            for i in range(len(pred_bbx)):                    
                # if iou threshold is greater than 50% we have a hit
                iou_val = iou(box, pred_bbx[i])
                if iou_val >= iou_thre:
                    pred_voter.append(pred_species[i])
                    
                    if save_to is not None:
                        # save predicted box: 
                        #tile name + bird index (true) + pred index + voter index + pred label + true label
                        tname = os.path.basename(file)[:-len(img_ext)]
                        vi = len(pred_voter)-1
                        plab = pred_species[i]
                        tlab = lab_cls2cat[annt_lab[birds]]
                        save_name = os.path.join(
                            save_to, f'{tname}bi{birds:04d}.pi{i:04d}.vi{vi:04d}.plab_{plab}.tlab_{tlab}.{img_ext}')

                        x1 = int(max(np.floor(pred_bbx[i][0]), 0))
                        y1 = int(max(np.floor(pred_bbx[i][1]) , 0))
                        x2 = int(min(np.ceil(pred_bbx[i][2]), im.shape[0]))
                        y2 = int(min(np.ceil(pred_bbx[i][3]), im.shape[1]))
                        im_pred_box = im[y1:y2, x1:x2]
                        cv2.imwrite(save_name, im_pred_box)
                        
            if len(pred_voter):
                if 0:#1 in pred_voter:
                    pred_total.append(1)  # ugly hack but we want more SAT!
                else:
                    pred_total.append(max(pred_voter,key=pred_voter.count))
                    
            # if bird is not found then we append an extra category saying it was not found
            try:
                cid = annt[birds]['category_id']
            except:
                cid = lab_cls2cat[annt_df.loc[birds,'class_id']]
                print(birds, annt)
            truth_total.append(cid)
            if not len(pred_voter):
                pred_total.append(len(bird_species))
    return pred_total, truth_total


def plot_confusion_matrix(cm, target_names, cmap = None, normalize = True, figure = None):
    """
    Given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if figure is None:
        fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi=200)
    else:
        fig, ax = figure
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(im, cax=cax, orientation='vertical')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks, target_names, rotation=45, ha="right", rotation_mode="anchor")
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
    ax.set_xlabel(f'Predicted label')#\naccuracy={accuracy:0.2f}; misclass={accuracy:0.2f}')
    