#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


import json
import os, sys, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import random
from tqdm.auto import tqdm
import torch
from datetime import date
import shutil

from utils.augmentation import dataset_aug
from utils.dataloader import register_datasets
from utils.hyperparameter import main_hyper, main_fit
from utils.cropping import crop_dataset, train_val_test_split, correct_labels


"""
Your dataset should be organized in the following format:

-- curr_dir/         (This is your project root directory)
    |-- ...
    |__ data/        (This is the dir to store your datasets)
        |-- ...            
        |__ 22S/     (This is the dataset that you will use to train a model in this script)
            |__ raw/ (This should contain the raw large image and corresponding annotation files)
                |-- 10k x 10k images [jpeg]
                |-- annotation files [csv]
    
"""
root = os.getcwd()
data_root = root + "/data/22F"
raw_dir = data_root + "/raw"
##################################################################################################################
# clean the annotation labels: D2K 30 Codes 20221013.xlsx
# notes = pd.read_excel("./data/D2K 30 Codes 20221013.xlsx")
# correct_labels(raw_dir)

################################################################################################################
# only if we are cropping ourselves

# make the cropped dataset
tile_dir = data_root + "/tiled"
if not os.path.exists(tile_dir):
    os.makedirs(tile_dir, exist_ok=True)

    #performing the cropping
    crop_dataset(raw_dir, tile_dir, annot_file_ext = 'csv', crop_height = 640, crop_width = 640)
    
#################################################################################################################
# make the 3 directories
crop_dir = data_root + "/split" #'./model_dataset'
if not os.path.exists(crop_dir):
    os.makedirs(crop_dir, exist_ok=True)

    train_val_test_split(tile_dir, crop_dir, train_frac=0.7, val_frac=0.15, seed=4)

# This is to change to bbx files into csv
dirs = [d for d in os.listdir(crop_dir) if not d.startswith('.') and not d.startswith('_')]

# ************** Optional: Printing the distribution of the birds in each dataset ************
fig, axes = plt.subplots(1,3,figsize=(15,10))
target_data = {}
for ax,d in zip(axes, dirs):
    target_data[d] = []
    for f in glob.glob(os.path.join(crop_dir, d, '*.csv')):
        target_data[d].append(pd.read_csv(f, header=0,
                                       names=["class_id", "class_name", "x", "y", "width", "height"]))
    target_data[d] = pd.concat(target_data[d], axis=0, ignore_index=True)

    # Visualize dataset
    id_counts = target_data[d]["class_id"].value_counts()
    ax.set_title(f'\n {d} - Bird Species Distribution ({len(id_counts)})')
    id_counts[::-1].plot.barh(ax=ax)
#     ax.set_xscale('log')
    
# ************** determine which bird species to train on *************************************
# populating the species for training (that have more than 10 images)
id_count = target_data["Validate"]["class_id"].value_counts()
BIRD_SPECIES = id_count.loc[id_count >= 20].index.values


# keep species that are in all train, val, and test sets
BIRD_SPECIES = np.intersect1d(
    BIRD_SPECIES,
    np.intersect1d(
        target_data["Validate"]["class_id"].unique(),
        target_data["Test"]["class_id"].unique()
    )
)

BIRD_SPECIES = ['ROTE', 'SANE'] # Hard coded for now due to type error

print(len(BIRD_SPECIES), BIRD_SPECIES)
# populating the species map
SPECIES_MAP = {}
for i, bird in enumerate(BIRD_SPECIES):
    SPECIES_MAP[i] = bird

print(SPECIES_MAP)
birds_species_names = BIRD_SPECIES

# ************** the data augmentation part in this section of the code *************************************
# this data augmentation code only works on the training set!!!
# the output direction is "aug_dir"
# dst_dir is the folder of training data(only after cropping)

# dst_dir = crop_dir + '/Train/'
# aug_dir is where we put image after doing data augmentation
# aug_dir = data_root + "temp/"
# if not os.path.exists(aug_dir):
#     os.makedirs(aug_dir, exist_ok=True)
#
#     # Minimum portion of a bounding box being accepted in a subimage
#     overlap = 0.3
#
#     # List of species that we want to augment (PLEASE include the full name)
#     minor_species = [s for s in BIRD_SPECIES if s not in ['ROT', 'LAGUA', 'SAT', 'MTRNA', 'OTHRA']]
#
#     # Threshold of non-minor creatures existing in a subimage
#     thres = .4
#
#     # [horizontal filp, vertical flip, left rotate, right rotate, [brightness/contrast tunning, number of images produced]]
#     aug_command = [1, 1, 1, 0, [0, 2]] #[1, 1, 1, 0, [1, 2]]
#
#     dataset_aug(dst_dir, aug_dir, minor_species, overlap, thres, aug_command, img_ext='JPEG', annot_file_ext='csv',
#                 crop_height=640, crop_width=640)
#
#     # copy files from aug_list(certain files in aug_dir) to dst_dir (train data set)
#     aug_list = glob.glob(os.path.join(aug_dir, '*'))
#     for i in aug_list:
#         shutil.copy2(i, dst_dir)


#########################################################################################################################
# registering the data in detectron2
img_ext = '.JPEG'
dirs_full = [os.path.join(crop_dir, d) for d in os.listdir(crop_dir) if not d.startswith('.') and not d.startswith('_')]

# Bird species used by object detector. Species contained in dataset that are
# not contained in this list will be categorized as an "Unknown Bird"

# Bounding box colors for bird species (used when plotting images)
NUM_COLORS = len(BIRD_SPECIES)
cm = plt.get_cmap('gist_rainbow')
BIRD_SPECIES_COLORS = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

register_datasets(dirs_full, img_ext, BIRD_SPECIES, bird_species_colors=BIRD_SPECIES_COLORS, unknown_bird_category=True)

########################################################################################################################
# training the bird species model using Faster R-CNN
torch.cuda.empty_cache()

# Weight for loss function (Unknown class should not be included in dist)
dist = target_data["Train"]["class_id"].value_counts()
custom_weight = (np.log(max(dist)/dist) + 1).to_list()

# name of the model output: {retinanet, faster_rcnn}
MODEL = 'faster_rcnn'
model_output_dir = f'./output/Training_models/06_22_bay_tune_retina_{len(BIRD_SPECIES)}class_set_I_aug'

if MODEL == 'faster_rcnn' :
    cfg_parms = {'NUM_WORKERS': 0, 'IMS_PER_BATCH': 8, 'BASE_LR': .01, 'GAMMA': 0.001,
                 'WARMUP_ITERS': 1, 'MAX_ITER': 1500,
                 'STEPS': [899], 'CHECKPOINT_PERIOD': 899, 'output_dir': model_output_dir,
                 'model_name': "faster_rcnn_R_50_FPN_1x", 'BIRD_SPECIES': BIRD_SPECIES, 'Custom': True,
                 'weight': custom_weight}
elif MODEL == 'retinanet':
    cfg_parms = {'NUM_WORKERS': 0, 'IMS_PER_BATCH': 4, 'BASE_LR': .001, 'GAMMA': 0.01,
                 'WARMUP_ITERS': 1, 'MAX_ITER': 1500,
                 'STEPS': [899], 'CHECKPOINT_PERIOD': 899, 'output_dir': model_output_dir,
                 'model_name': "retinanet_R_50_FPN_1x", 'BIRD_SPECIES': BIRD_SPECIES, 'Custom': False}
else:
    raise    
tuned_cfg_params = cfg_parms

# # hyperparameter tuning
# tuned_cfg_params = main_hyper(cfg_parms, iterations=20)  # Bayesian iters
# tuned_cfg_params['MAX_ITER'] = tuned_cfg_params['MAX_ITER'] + 100
tune_weight_dir, loss = main_fit(tuned_cfg_params)

