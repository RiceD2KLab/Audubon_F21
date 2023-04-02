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
from ..const import COL_NAMES, SAVE_FIG, DPI

#######################################################################################
# Data visualization

def get_cmap(num, name='tab20c'):
    '''
    Return a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    '''
    return plt.cm.get_cmap(name, num)

def plot_distribution(data_frame, col_name, info, path, filt=None):
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
    chart = axs.bar(idx_list, val_list, color=color_list)
    axs.set_title(title)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.set_xticks(range(len(idx_list)))
    axs.set_xticklabels(idx_list, rotation=45, ha='right', fontsize=12)
    axs.set_ylim(0, val_counts.max() * 1.2)
    for i in range(len(chart)):
        axs.text(i, chart[i].get_height() + 0.5, chart[i].get_height(), ha='center', fontsize=10)
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
    height, width, n_channels = image.shape
    print(image.shape)
    # num_row, num_col, dummy_channel = image.shape
    annos = csv_to_df(bbx_name, COL_NAMES).to_dict()

    # plot image
    fig, axs = plt.subplots(figsize=(width / DPI, height / DPI), dpi=DPI)
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
    """ 
    Plot train and test loss for each training epoch of a model.
    
    Input:
        train_loss : A list of training losses for each epoch. 
        test_loss: A list of test losses for each epoch. 
        path : A path to save the plot.
        title: The title of the plot (string).
    
    Output:
        A line chart illustrating the train and test loss for each training epoch.
    """
    fig, axs = plt.subplots()
    axs.plot(train_loss, label="Training loss")
    axs.plot(test_loss, label="Test loss")
    axs.set_xlabel("Number of epochs")
    axs.set_ylabel("Loss")
    axs.set_title(title)
    axs.legend()

    if SAVE_FIG:
        fig.savefig(path + title + '.pdf', bbox_inches='tight')

    return fig

def plot_precision_recall(stat_arr, epochs, path, title):
    """ 
    Plot the precision and recall at different IoU thresholds for each training epoch of a model.  
    
    Input:
        stat_arr: An array containing the precision and recall values for IoU thresholds of 0.5 and 0.75 for each epoch. 
        epochs: A list of epoch numbers.
        path: The path to save the plot (string).
        title: The title of the plot (string). 
    """
    fig, axs = plt.subplots()
    axs.plot(epochs, stat_arr[:, 0], label="Precision with IoU=0.5")
    axs.plot(epochs, stat_arr[:, 1], label="Precision with IoU=0.75")
    axs.plot(epochs, stat_arr[:, 2], label="Recall with IoU=0.5")
    axs.plot(epochs, stat_arr[:, 3], label="Recall with IoU=0.75")
    axs.set_xlabel("Number of epochs")
    axs.set_ylabel("Precision and recall")
    axs.set_title(title)
    axs.legend()

    if SAVE_FIG:
        fig.savefig(path + title + '.pdf', bbox_inches='tight')

    return fig

def show(img):
    ''' 
    Show an image.
    
    Input:
        img: A Torch tensor representing an image.
        
    Outputs:
        fig: The figure object displaying the image. 
    '''
    n_channels, height, width = img.shape
    fig, axs = plt.subplots(figsize=(width / DPI, height / DPI), dpi=DPI)
    img = img.detach()
    img = F.to_pil_image(img)
    axs.imshow(np.asarray(img))
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig

def visualize_predictions(file_paths, output, path, title, score_threshold=0.8):
    ''' 
    Visualize bounding box predictions for the test dataset.
    
    Input:
        file_paths: A list of file paths (string) to the images in the test dataset.
        output: A dictionary containing the predictions made by the model. 
        path: The directory where the output image should be saved (string).
        title: The title of the output image (string).
        score_threshold: The minimum score threshold for drawing bounding boxes. Defaults to 0.8. 
        
    Output:
        fig: Each image overlaid with predicted bounding boxes. 
    '''
    img = read_image(file_paths)
    result = draw_bounding_boxes(img, output['boxes'][output['scores'] > score_threshold],
                                 colors='blue', width=5)
    fig = show(result)
    if SAVE_FIG:
        fig.savefig(path + title + '.jpg', bbox_inches='tight')

def plot_confusion_matrix():
    ''' Plot a confusion matrix '''
    pass
