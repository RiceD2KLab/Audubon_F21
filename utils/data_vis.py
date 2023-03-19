''' 
Data visualization 
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from .data_processing import csv_to_df
from ..const import COL_NAMES, SAVE_FIG

#######################################################################################
# Data visualization

def get_cmap(num, name='tab20c'):
    '''
    Return a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    '''
    return plt.cm.get_cmap(name, num)

def plot_distribution(data_frame, col_name, 
                      info, path, filt=None):
    ''' 
    Plot a barchart of the value counts of a column in a dataframe.
    
    Input:
        data_frame : Pandas dataframe containing the column to plot.
        col_name : The name of the column to plot.
        info : A tuple containing the x-label, y-label, and title of the plot.
        path : The directory in which to save the plot.
        filt : int or None. If not None, only show categories with a count of at least `filt`.
    
    Output:
        A barchart of the value counts for the specified column. 
    '''
    x_label, y_label, title = info
    plt.rcdefaults()
    val_counts = data_frame[col_name].value_counts()
    # print(val_counts)
    if filt:
        val_counts = val_counts[val_counts >= filt]
    idx_list = val_counts.index.tolist()
    val_list = val_counts.values.tolist()
    cmap = get_cmap(len(idx_list))
    color_list = [cmap(idx) for idx in range(len(idx_list))]

    # make plot
    fig, axs = plt.subplots(figsize=(10, 6))
    chart = axs.barh(idx_list, val_list, color=color_list)
    axs.set_title(title)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.invert_yaxis()
    axs.bar_label(chart)
    if SAVE_FIG:
        fig.savefig(path + title + '.pdf', bbox_inches='tight')
    return fig

def plot_boxes(jpg_name, bbx_name, title, path):
    ''' 
    Plot an image overlaid with annotation boxes.
    
    Input:
        jpg_name : The filename of the image file to be plotted.
        bbx_name : The filename of the CSV file containing the bounding box coordinates.
        title : The title of the plot.
        path : The path where the resulting image file will be saved.
    Output:
        A plot of the image overlaid with annotation boxes.
    '''
    image = plt.imread(jpg_name)
    # num_row, num_col, dummy_channel = image.shape
    annos = csv_to_df(bbx_name, COL_NAMES).to_dict()

    # plot image
    fig, axs = plt.subplots()  
    axs.imshow(image, origin='lower')
    axs.set_axis_off()
    # axs.set_title(title)
    # draw bounding boxes (rectangles)
    for idx in range(len(annos["x"])):
        rect = patches.Rectangle((annos["x"][idx], annos["y"][idx]), 
            annos["width"][idx], annos["height"][idx], 
            linewidth=0.5, edgecolor='b', facecolor='none')
        axs.add_patch(rect)

    if SAVE_FIG:
        fig.savefig(path + title + '.jpg', bbox_inches='tight')

    return fig

def plot_training_curves(train_loss, test_loss, path, title):
    """ Plot regression train and test loss """
    fig, axs = plt.subplots()
    axs.plot(train_loss, label="Training loss")
    axs.plot(test_loss, label="Test loss")
    axs.set_xlabel("Number of epochs")
    axs.set_ylabel("Loss")
    axs.legend()

    if SAVE_FIG:
        fig.savefig(path + title + '.pdf', bbox_inches='tight')

    return fig

def show(imgs):
    ''' Show a list of images '''
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for idx, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, idx].imshow(np.asarray(img))
        axs[0, idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig

def visualize_predictions(file_paths, output, path, title, score_threshold=0.8):
    ''' Visualize predictions for the test dataset '''
    fig = read_image(file_paths)
    result = draw_bounding_boxes(fig, output['boxes'][output['scores'] > score_threshold],
                                 colors='blue', width=5)
    fig = show(result)
    if SAVE_FIG:
        fig.savefig(path + title + '.jpg', bbox_inches='tight')

def plot_confusion_matrix():
    ''' Plot a confusion matrix '''
    pass