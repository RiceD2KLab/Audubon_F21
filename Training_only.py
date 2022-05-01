import os, sys, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import cv2
from tqdm.autonotebook import tqdm
import torch

# Data Directory name
crop_dir = 'C://Users\VelocityUser\Documents\D2K TDS D\TDS D-10'

# ######################################################################################################################


# This is to change to bbx files into csv
dirs = os.listdir(crop_dir)

##################################write all the bbx files into csv files#################################################
# from utils.cropping_hank import csv_to_dict, dict_to_csv
#
# for d in dirs:
#     for f in glob.glob(os.path.join(crop_dir, d, '*.bbx')):
#         dict_bird = csv_to_dict(f, annot_file_ext='bbx')
#         # print(dict_bird)
#         dict_to_csv(dict_bird, os.path.split(f)[0], empty=False, img_ext='bbx')
# #
# #######################################################################################################################
# # # ************** the data augmenation part in this section of the code *************************************
# # # this data augmentation code only works on the training set!!!
# # # the output direction is "aug_dir"
# #
import shutil
from utils.augmentation import AugTrainingSet, dataset_aug

# # dst_dir is the folder of training data(only after cropping)
dst_dir = crop_dir + '/Train'
# aug_dir is where we put image after doing data augmentation
os.makedirs('C://Users\\VelocityUser\\Documents\\Audubon_F21\\temp', exist_ok=True)
aug_dir = 'C://Users\\VelocityUser\\Documents\\Audubon_F21\\temp'
#
#
# # Minimum portion of a bounding box being accepted in a subimage
overlap = 0.2
#
# # List of species that we want to augment (PLEASE include the full name)
minor_species = ["REEGA","WHIBA","ROSPA","SATEA", "BRPEA", "TRHEA"]
#
# # Threshold of non-minor creatures existing in a subimage
thres = .3

#[horizontal filp, vertical flip, left rotate, right rotate, [brightness/contrast tunning, number of images produced]]
aug_command = [1,1,1,0,[1,2]]

# AugTrainingSet(dst_dir, aug_dir, minor_species, overlap, thres, img_ext='JPG', annot_file_ext='csv')
# dataset_aug(dst_dir, aug_dir, minor_species, overlap, thres,
#             aug_command, img_ext = 'JPG',annot_file_ext='csv',crop_height=640, crop_width=640)
# #
# #
# """"""
# aug_list = glob.glob(os.path.join(aug_dir, '*'))
# # print(aug_list)
# # #
# # print(dst_dir)
#
# for i in aug_list:
#
#     shutil.copy2(i, dst_dir)  # copy files from aug_list(certain files in aug_dir) to dst_dir (train data set)
#     # print(i)

############################Printing the distribution of the birds in each dataset######################################
for d in dirs:
    target_data = []
    for f in glob.glob(os.path.join(crop_dir, d, '*.csv')):
        target_data.append(pd.read_csv(f, header=0,
                                       names=["class_id", "class_name", "x", "y", "width", "height"]))
    target_data = pd.concat(target_data, axis=0, ignore_index=True)

    # Visualize dataset
    print(f'\n {d} - Bird Species Distribution')
    print(target_data["class_name"].value_counts())
    print('\n')

# #
# # ########################################################################################################################
# # # registering the data in detectron2
# #
from utils.dataloader import register_datasets

data_dir = crop_dir
# img_ext = '.JPEG'
img_ext = '.JPG'
dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]

# Bird species used by object detector. Species contained in dataset that are
# not contained in this list will be categorized as an "Unknown Bird"
BIRD_SPECIES = ["BCNHA", "BLSKA", "BRPEA", "LAGUA", "REEGA",
                'ROSPA', 'ROTEA', 'SATEA', 'TRHEA',
                'WHIBA']

birds_species_names = BIRD_SPECIES

SPECIES_MAP = {0: 'BCNHA', 1: 'BLSKA', 2: 'BRPEA',
               3: 'LAGUA', 4: 'REEGA', 5: 'ROSPA',
               6: 'ROTEA', 7: 'SATEA', 8: 'TRHEA', 9: 'WHIBA'}

# #
# # # Bounding box colors for bird species (used when plotting images)
BIRD_SPECIES_COLORS = [(255, 0, 0), (255, 153, 51), (0, 255, 0),
                       (0, 0, 255), (255, 51, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]
#
register_datasets(dirs, img_ext, BIRD_SPECIES, bird_species_colors=BIRD_SPECIES_COLORS, unknown_bird_category=False)

#####################################################################################################
# target_data = []
# for f in glob.glob(os.path.join(crop_dir, 'Train', '*.csv')):
#     target_data.append(pd.read_csv(f, header=0,
#                                    names=["class_id", "class_name", "x", "y", "width", "height"]))
# target_data = pd.concat(target_data, axis=0, ignore_index=True)
#
# # Visualize dataset
# print('- Bird Species Distribution')
# print(target_data["class_name"].value_counts())
# print('\n')
#
# va = target_data["class_name"].value_counts()
# data_name = va.index.to_list()
# data_val = va.to_numpy()
# weight = np.zeros((len(BIRD_SPECIES), 1))
#
# total_bird = np.sum(data_val)
#
# for indx, bird in enumerate(BIRD_SPECIES):
#     for indy, j in enumerate(data_name):
#         if bird == j:
#             weight[indx] = 1 - data_val[indy] / total_bird
# # # # #######################################################################################################################
# # # #
# # # #
# # # # ########################################################################################################################
# # # # training the bird species model using Faster R-CNN
torch.cuda.empty_cache()

custom_weight = []

#name of the model output
model_output_dir = '../Training_models/04_25_10class_aug_T2'

cfg_parms = {'NUM_WORKERS': 0, 'IMS_PER_BATCH': 6, 'BASE_LR': .001, 'GAMMA': 0.01,
             'WARMUP_ITERS': 1, 'MAX_ITER': 800,
             'STEPS': [499], 'CHECKPOINT_PERIOD': 499, 'output_dir': model_output_dir,
             'model_name': "faster_rcnn_R_50_FPN_1x", 'BIRD_SPECIES': BIRD_SPECIES, 'Custom': False}

from utils.hyperparameter import main_hyper, main_fit

# hyperparameter tunning
tuned_cfg_params = main_hyper(cfg_parms, iterations=40)
tuned_cfg_params['MAX_ITER'] = tuned_cfg_params['MAX_ITER'] + 100
tune_weight_dir, loss = main_fit(tuned_cfg_params)