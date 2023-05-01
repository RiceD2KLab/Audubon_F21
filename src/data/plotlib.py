import matplotlib.pyplot as plt
import os
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_cmap(num, name='tab20c'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color.

    Args:
        num (int): The number of colors to return.
        name (str): The name of the color map. Default is 'tab20c'.

    Returns:
        A matplotlib color map function.
    '''
    return plt.cm.get_cmap(name, num)


def plot_distribution(data_frame, col_name,
                      x_label, y_label, title, path=None, filt=None):
    '''
    Plots a barchart of the value counts of a column in a dataframe.

    Args:
        data_frame (pandas dataframe): The dataframe containing the column to plot.
        col_name (str): The name of the column to plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
        path (str): The directory to save the plot. Default is None.
        filt (int): The minimum count of categories to include. Default is None.

    Returns:
        A matplotlib figure of the value counts for the specified column.
    '''

    plt.rcdefaults()
    val_counts = data_frame[col_name].value_counts()

    # Filter value counts if minimum count is specified
    if filt:
        val_counts = val_counts[val_counts >= filt]
    idx_list = val_counts.index.tolist()
    val_list = val_counts.values.tolist()

    # Get a colormap for the plot
    cmap = get_cmap(len(idx_list))
    color_list = [cmap(idx) for idx in range(len(idx_list))]

    # Create plot
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

    # Save plot if path is specified
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + title + '.pdf', bbox_inches='tight')
    return fig


def plot_curves(arr1, arr2, label1, label2, xlabel, ylabel, title, path=None):
    '''
    Plots two curves on the same plot.

    Args:
        arr1 (array): An array of values to plot for the first curve.
        arr2 (array): An array of values to plot for the second curve.
        label1 (str): The label for the first curve.
        label2 (str): The label for the second curve.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the plot.
        path (str): The directory to save the plot. Default is None.

    Returns:
        A matplotlib figure of the two curves.
    '''
    # Create plot with two curves
    fig, axs = plt.subplots()
    axs.plot(arr1, label=label1)
    axs.plot(arr2, label=label2)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    axs.legend()

    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + title + '.pdf', bbox_inches='tight')

    return fig


def plot_precision_recall(stat_arr, xlabel, ylabel, title, path=None):
    '''
    Plots the precision and recall at different IoU thresholds for each training epoch of a model.

    Args:
        stat_arr (ndarray): An array containing the precision and recall values for IoU thresholds of 0.5 and 0.75 for each epoch.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the plot.
        path (str): The directory to save the plot. Default is None.

    Returns:
        A matplotlib figure of the precision and recall at different IoU thresholds for each epoch.
    '''
    fig, axs = plt.subplots()

    # Plot precision and recall for each epoch for IoU thresholds of 0.5 and 0.75
    axs.plot(stat_arr[:, 0], label="Precision with IoU=0.5")
    axs.plot(stat_arr[:, 1], label="Precision with IoU=0.75")
    axs.plot(stat_arr[:, 2], label="Recall with IoU=0.5")
    axs.plot(stat_arr[:, 3], label="Recall with IoU=0.75")
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    axs.legend()

    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + title + '.pdf', bbox_inches='tight')

    return fig


def show(img, dpi):
    '''
    Shows an image.

    Args:
        img (tensor): The image tensor.
        dpi (int): The dots per inch.

    Returns:
        A matplotlib figure of the image.
    '''
    n_channels, height, width = img.shape

    # Create a figure with the appropriate dimensions and dots per inch
    fig, axs = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    
    # Convert the image tensor to PIL image and display it in the figure
    img = img.detach()
    img = F.to_pil_image(img)
    axs.imshow(np.asarray(img))
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig


def visualize_predictions(file_paths, output, path, title, dpi, score_threshold=0.8):
    '''
    Visualize bounding box predictions for the test dataset.

    Args:
        file_paths (list): A list of file paths (string) to the images in the test dataset.
        output (dict): A dictionary containing the predictions made by the model.
        path (str): The directory where the output image should be saved.
        title (str): The title of the output image.
        dpi (int): The DPI of the output image.
        score_threshold (float): The minimum score threshold for drawing bounding boxes. 
            Defaults to 0.8.

    Returns:
        A figure object containing each image overlaid with predicted bounding boxes.
    '''
    img = read_image(file_paths)

    # Draw the predicted bounding boxes on the image
    result = draw_bounding_boxes(img, output['boxes'][output['scores'] > score_threshold],
                                 colors='blue', width=3)
    fig = show(result, dpi)
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + title + '.jpg', bbox_inches='tight')
    return fig


def plot_confusion_matrix(true_label_list, predicted_list, class_names, title='Confusion matrix', path=None):
    '''
    Plot a confusion matrix.

    Args:
        true_label_list (list): A list of true labels.
        predicted_list (list): A list of predicted labels.
        class_names (list): A list of class names.
        title (str): The title of the output image. Defaults to 'Confusion matrix'.
        path (str): The directory where the output image should be saved.

    Returns:
        A figure object containing the confusion matrix.
    '''

    # Caculate the confusion matrix and create a display for it
    conf_mat = confusion_matrix(true_label_list, predicted_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)

    # Plot the display in a figure
    fig, ax = plt.subplots(figsize=(25, 10))
    disp.plot(ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + title + '.pdf', bbox_inches='tight')

    return fig
