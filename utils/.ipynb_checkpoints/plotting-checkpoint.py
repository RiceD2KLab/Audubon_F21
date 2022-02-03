import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def plot_img_bbx(image, annotation_lst, color_dict=None):
    """
    This is a plotting function to check if the bndbox annotations convertion is done correctly
    INPUT:
    image -- PIL image module to be
    annotation_lst -- a list containing all the annotations for this image
    color_dict -- a dictionary containing color scheme to be used to outline bounding boxes for each class
    """
    w, h = image.size
    plotted_img = ImageDraw.Draw(image)

    for annot in annotation_lst:
        obj_class, desc, x_min, y_min, x_max, y_max = annot
        if color_dict:
            plotted_img.rectangle(((x_min, y_min), (x_max, y_max)), width=5, outline=color_dict[obj_class])
        else:
            plotted_img.rectangle(((x_min, y_min), (x_max, y_max)), width=5)
        plotted_img.text((x_min, y_min - 10), obj_class)

    plt.figure(figsize=(20, 10)) # this is for AWS imaging, 60,30 is too big for AWS sagemaker
#     plt.figure(figsize=(60, 30)) 
    plt.imshow(np.array(image), interpolation='nearest')
    plt.axis('off')
    plt.show()