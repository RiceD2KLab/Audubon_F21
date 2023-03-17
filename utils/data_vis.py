''' 
Data visualization 
'''

import matplotlib.pyplot as plt
from matplotlib import patches
from audubon.utils.data_processing import csv_to_df
from audubon.const import COL_NAMES, SAVE_FIG, DPI

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
        fig.savefig(path + title + '.pdf')
    return fig

def plot_boxes(jpg_name, bbx_name, title, path, show=False):
    ''' 
    Plot an image overlaid with annotation boxes.
    
    Input:
        jpg_name : The filename of the image file to be plotted.
        bbx_name : The filename of the CSV file containing the bounding box coordinates.
        title : The title of the plot.
        path : The path where the resulting image file will be saved.
        show : Whether to plot the image. Default is no (False).
    Output:
        A plot of the image overlaid with annotation boxes.
    '''
    image = plt.imread(jpg_name)
    num_row, num_col, dummy_channel = image.shape
    annos = csv_to_df(bbx_name, COL_NAMES).to_dict()

    # plot image
    fig, axs = plt.subplots(figsize=(num_col / DPI, num_row / DPI), dpi=DPI)  
    axs.imshow(image, origin='lower')
    axs.set_axis_off()
    axs.set_title(title)
    # draw bounding boxes (rectangles)
    for idx in range(len(annos["x"])):
        rect = patches.Rectangle((annos["x"][idx], annos["y"][idx]), 
            annos["width"][idx], annos["height"][idx], 
            linewidth=0.5, edgecolor='b', facecolor='none')
        axs.add_patch(rect)

    if SAVE_FIG:
        fig.savefig(path + title + '.jpg')
    if show == True:
        fig.show()
    return fig
