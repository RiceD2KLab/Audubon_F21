import pandas as pd
import os
import numpy as np


def csv_to_df(csv_path):
    """
    Reads a CSV file and returns a pandas dataframe.

    Args:
        csv_path (str): The path of the CSV file.

    Returns:
        A pandas dataframe.
    """
    df = pd.read_csv(csv_path, header=0)
    return df


def concat_frames(file_names):
    """
    Concatenates dataframes from a list of CSV file names into a single pandas dataframe.

    Args:
        file_names (list of str): A list of CSV file names to read and concatenate.

    Returns:
        A pandas dataframe containing rows from all the concatenated CSV files.
    """
    # Get a list of Panda dataframes from a list of csv files
    frames = [csv_to_df(file_name) for file_name in file_names]
    return pd.concat(frames, axis=0, ignore_index=True)


def get_file_names(path, extension):
    """
    Returns a sorted list of file names that match the specified file extension in the given directory.

    Args:
        path (str): The directory path to search for files.
        extension (str): The file extension to match. For example, '.csv' for annotations or '.jpg' for images.

    Returns:
        A sorted list of file names that match the specified file extension.
    """
    file_names = []

    # Get file names of files with the correct extnesion
    for file in os.listdir(path):
        if file.endswith(extension):
            file_names.append(os.path.join(path, file))
    return sorted(file_names)


def split_img_annos(img_files, anno_files, frac, seed=None):
    """
    Splits the image and annotation files into three sets (training, testing, and validation) based on a given fraction.

    Args:
        img_files (list of str): A list of image file paths.
        anno_files (list of str): A list of annotation file paths.
        frac (tuple): A tuple of three floats that specifies the fraction of data
            to be used for training, testing, and validation.
        seed: A random seed to use for shuffling the files. Defaults to None.

    Returns:
        A list of three dictionaries, one for training, testing, and validation respectively,
        where each dictionary contains the following keys:
            - 'jpg': a list of image file paths for the set.
            - 'csv': a list of annotation file paths for the set.
    """
    if seed:
        np.random.seed(seed)

    num_of_files = len(img_files)

    # calculate the indices for the training, testing, and validation sets based on the fraction
    train_idx = int(num_of_files * frac[0])
    test_idx = train_idx + int(num_of_files * frac[1])
    val_idx = min(test_idx + int(num_of_files * frac[2]), num_of_files)

    # shuffle the indices to randomly split the dataset
    indices = np.arange(num_of_files)
    np.random.shuffle(indices)

    # divide the files into training, testing, and validation sets based on the calculated indices
    train_test_val = [
        {'jpg': [img_files[idx] for idx in indices[:train_idx]],
         'csv': [anno_files[idx] for idx in indices[:train_idx]]},
        {'jpg': [img_files[idx] for idx in indices[train_idx: test_idx]],
         'csv': [anno_files[idx] for idx in indices[train_idx: test_idx]]},
        {'jpg': [img_files[idx] for idx in indices[test_idx:val_idx]],
         'csv': [anno_files[idx] for idx in indices[test_idx:val_idx]]},
    ]

    return train_test_val
