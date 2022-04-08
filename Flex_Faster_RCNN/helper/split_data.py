import os
import random
import torch
import numpy as np


def main():
    # random.seed(0)  # random seeds

    # input: annotation xml files

    files_path = 'C://Users\\VelocityUser\\Documents\\D2K TDS A\\6_class_combine\\images'
    # files_path = "/Users/maojietang/Downloads/VOCdevkit/VOC2012/Annotations"
    # assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.1

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path) if file.split(".")[0] != ''])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:

        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


def train_split(files_path, val_size, param_dir):
    # assume val_size and test_size is the same

    # random.seed(0)  # random seeds

    # input: annotation xml files

    files_name = np.array(sorted([file.split(".")[0] for file in os.listdir(files_path) if file.split(".")[0] != '']))
    files_num = len(files_name)

    file_indexs = files_name[torch.randperm(files_num).numpy()]

    train_files = []
    val_files = []
    test_files = []

    val_split = np.floor((val_size * files_num))

    for index, file_name in enumerate(file_indexs):
        if index < val_split:
            val_files.append(file_name)
        elif index < 2 * val_split:
            test_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_path = os.path.join(param_dir, 'train.txt')
        test_path = os.path.join(param_dir, 'test.txt')
        val_path = os.path.join(param_dir, 'val.txt')
        if os.path.exists(train_path) or os.path.exists(test_path) or os.path.exists(val_path):
            os.remove(train_path)
            os.remove(test_path)
            os.remove(val_path)
        train_f = open(train_path, "x")
        eval_f = open(val_path, "x")
        test_f = open(test_path, "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
        test_f.write("\n".join(test_files))

    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
    print('Data processing completed')
