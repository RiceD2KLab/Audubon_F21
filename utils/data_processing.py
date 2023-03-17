''' 
Data processing.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######################################################################################
# Data processing

def get_file_names(path, extension):
    ''' 
    Return a sorted list of file names that match the specified file extension in the given directory.
    
    Input:
        path : Directory path to search for files (in string format).
        extension : File extension to match (in string format). In our case, .csv for annotations or .jpg for images.
    Output:
        A sorted list of file names that match the specified file extension.
    '''
    file_names = []
    for file in os.listdir(path):
        if file.endswith(extension):
            file_names.append(os.path.join(path, file))
    return sorted(file_names)

def csv_to_df(file_name, cols):
    ''' 
    Read a CSV file and save it to a pandas dataframe.
    
    Input:
        file_name: Name of the CSV file to read (in string format).
        cols: List of column names to use for the dataframe.
    Output: 
        A pandas dataframe containing rows in the CSV file, and the first row is the header.
    '''
    data_df = pd.read_csv(file_name, header=0, names=cols)
    return data_df

def concat_frames(file_names, cols):
    ''' 
    Concatenate dataframes saved in a list of file names into a single pandas dataframe. 
    
    Input:
        file_names: A list of CSV file names (strings) to read and concatenate.
        cols: List of column names to use for the dataframe.
    Output:
        A pandas dataframe containing rows from all the concatenated CSV files
    ''' 
    frames = [csv_to_df(file_name, cols) for file_name in file_names]
    return pd.concat(frames, axis=0, ignore_index=True)

def read_jpg(file_name):
    ''' 
    Read a JPG file and save it to a numpy array.
    
    Input:
        file_name: A JPG file name (in string format).
    Output:
        A numpy array containing the RGB values of the JPG image.
    '''
    image = plt.imread(file_name)
    # print(image.shape)
    return image

def add_col(frame, added_col_name, col_name, values_dict):
    '''
    Add a column to a pandas dataframe by mapping values from an existing column using a dictionary. 
    
    Input:
        frame: An existing pandas dataframe.
        added_col_name: The name of the new column (string).
        col_name: The name of the existing column to map values from (string).
        values_dict: Dictionary that maps values from existing columns to values in new column.
    Output:
        Pandas dataframe with the new column added.
    '''
    frame[added_col_name] = frame[col_name].map(values_dict)
    return frame

def split_img_annos(img_files, anno_files, frac, seed=None):
    ''' 
    Split the image and annotation files into three sets (training, testing, and validation) based on a given fraction.
    Input:
        img_files : List of image file paths.
        anno_files : List of annotation file paths.
        frac : A tuple of three floats that specifies the fraction of data to be used for training, testing, and validation.
    Output:
        A list of three dictionaries, one for training, testing and validation respectively, where each dictionary contains the following keys:
            - 'jpg': a list of image file paths for the set.
            - 'csv': a list of annotation file paths for the set.
    '''
    if seed:
        np.random.seed(seed)

    num_of_files = len(img_files)
    train_idx = int(num_of_files * frac[0])
    test_idx = train_idx + int(num_of_files * frac[1])

    indices = np.arange(num_of_files)
    np.random.shuffle(indices)
    train_test_val = [
        {'jpg': [img_files[idx] for idx in indices[:train_idx]],
         'csv': [anno_files[idx] for idx in indices[:train_idx]]},
        {'jpg': [img_files[idx] for idx in indices[train_idx: test_idx]],
         'csv': [anno_files[idx] for idx in indices[train_idx: test_idx]]},
        {'jpg': [img_files[idx] for idx in indices[test_idx:]],
         'csv': [anno_files[idx] for idx in indices[test_idx:]]},
    ]

    return train_test_val

def coordinate_to_box(x_1, y_1, width, height):
    ''' 
    Convert object coordinates to bounding box coordinates
    
    Input:
        x_1: The x coordinate of the top left corner of the object (int)
        y_1: The y coordinate of the top left corner of the object (int)
        width: The width of the object (int)
        height: The height of the object (int)
    Output:
        A list containing bounding box coordinates [x_min, y_min, x_max, y_max]
    ''' 
    x_2, y_2 = x_1 + width, y_1 + height
    box = [x_1, y_1, x_2, y_2]

    return box
