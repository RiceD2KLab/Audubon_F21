import pandas as pd
import os
import numpy as np


def csv_to_df(csv_path):
    ''' Returns dataframe from csv file '''
    df = pd.read_csv(csv_path, header=0)
    return df


def concat_frames(file_names):
    '''
    Concatenate dataframes saved in a list of file names into a single pandas dataframe.

    Input:
        file_names: A list of CSV file names (strings) to read and concatenate.
        cols: List of column names to use for the dataframe.
    Output:
        A pandas dataframe containing rows from all the concatenated CSV files
    '''
    frames = [csv_to_df(file_name) for file_name in file_names]
    return pd.concat(frames, axis=0, ignore_index=True)


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


def split_img_annos(img_files, anno_files, frac, seed=None):
    '''
    Split the image and annotation files into three sets (training, testing, and validation) based on a given fraction.
    Input:
        img_files : List of image file paths.
        anno_files : List of annotation file paths.
        frac : A tuple of three floats that specifies the fraction of data
        to be used for training, testing, and validation.
    Output:
        A list of three dictionaries, one for training, testing and validation respectively,
        where each dictionary contains the following keys:
            - 'jpg': a list of image file paths for the set.
            - 'csv': a list of annotation file paths for the set.
    '''
    if seed:
        np.random.seed(seed)

    num_of_files = len(img_files)
    train_idx = int(num_of_files * frac[0])
    test_idx = train_idx + int(num_of_files * frac[1])
    val_idx = min(test_idx + int(num_of_files * frac[2]), num_of_files)

    indices = np.arange(num_of_files)
    np.random.shuffle(indices)
    train_test_val = [
        {'jpg': [img_files[idx] for idx in indices[:train_idx]],
         'csv': [anno_files[idx] for idx in indices[:train_idx]]},
        {'jpg': [img_files[idx] for idx in indices[train_idx: test_idx]],
         'csv': [anno_files[idx] for idx in indices[train_idx: test_idx]]},
        {'jpg': [img_files[idx] for idx in indices[test_idx:val_idx]],
         'csv': [anno_files[idx] for idx in indices[test_idx:val_idx]]},
    ]

    return train_test_val
