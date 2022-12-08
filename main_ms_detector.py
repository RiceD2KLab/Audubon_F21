import json
import os, sys, shutil, glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils.augmentation import dataset_aug
from utils.dataloader import register_datasets
from utils.hyperparameter import main_hyper, main_fit
from utils.cropping import crop_dataset, train_val_test_split, correct_labels

'''
    This script contains code for data preprocessing (tiling and augmentation) and multi-species bird 
    detector training. The code is partly adapted from previous Rice Audubon Teams' work. 

    Your dataset should be organized in the following format:

    -- curr_dir/         (This is your project root directory, where THIS script locates)
        |-- ...
        |-- main_ms_detector.py (* This script)
        |
        `-- data/        (This is the dir to store your datasets)
            |
            |-- D2K 30 Codes 20221013.xlsx  (Optional: Documentation of bird species code)
            |
            `-- 22All/     (This is the dataset that you will use to train a model in this script)
                |          Available/example options: 
                |              {22S: dataset for 22 spring team that has mixed tern labels (200 images);
                |               22F: new annotated terns-only dataset (10 images);
                |               22All: merges 22S and 22F by blacking-out (see details in our report)}
                |
                `-- raw/ (This should contain the large UAV images and corresponding annotation files)
                    |-- UAV images      [jpeg]
                    `-- annotation files [csv]
'''

data_root = './data/22F/'
#notes = pd.read_excel('./data/D2K 30 Codes 20221013.xlsx')

#########################################################
## I. Data processing: tiling, split, and augmentation ##
#########################################################

# clean the annotation labels
raw_dir = os.path.join(data_root, 'raw/')
correct_labels(raw_dir)

# make the tiled dataset and perform the cropping (if not already performed)
tile_dir = os.path.join(data_root, 'tiled/') 
if not os.path.exists(tile_dir):
    os.makedirs(tile_dir, exist_ok=True)
    crop_dataset(raw_dir, tile_dir, annot_file_ext = 'bbx', crop_height = 640, crop_width = 640)
    
# make dirs for train/val/test split 
split_dir = os.path.join(data_root, 'split/')
if not os.path.exists(split_dir):
    os.makedirs(split_dir, exist_ok=True)
    train_val_test_split(tile_dir, split_dir, train_frac = 0.7, val_frac = 0.15, seed = 0)

# format bbx files into csv (if needed)
dirs = [d for d in os.listdir(split_dir) if not d.startswith('.') and not d.startswith('_')]

# (optional) print the distribution of the birds in each dataset
fig, axes = plt.subplots(1, 3, figsize = (15,10))
target_data = {}
for ax,d in zip(axes, dirs):
    target_data[d] = []
    for f in glob.glob(os.path.join(split_dir, d, '*.csv')):
        target_data[d].append(
            pd.read_csv(f, header = 0, names = ['class_id', 'class_name', 'x', 'y', 'width', 'height'])
        )
    target_data[d] = pd.concat(target_data[d], axis = 0, ignore_index = True)

    # Visualize dataset
    id_counts = target_data[d]['class_id'].value_counts()
    ax.set_title(f'\n {d} - Bird Species Distribution ({len(id_counts)})')
    id_counts[::-1].plot.barh(ax=ax)
plt.savefig('datadist.png')

# determine on which bird species (e.g., that have more than 15 val images) to train
# please tune the threshold according to the data distributiona and to your need
id_count_thresh = 200  #multispecies=15; terns=200 
id_count = target_data['Validate']['class_id'].value_counts()
bird_spcs_use = id_count.loc[id_count >= id_count_thresh].index.values

# keep only species that are in all train, val, and test sets
bird_spcs_use = np.intersect1d(
    bird_spcs_use,
    np.intersect1d(
        target_data['Validate']['class_id'].unique(),
        target_data['Test']['class_id'].unique()
    )
)

# trash is not birds
bird_spcs_use = np.setdiff1d(bird_spcs_use, ['TRASH','OTHRA']).tolist()
print(len(bird_spcs_use), bird_spcs_use)

# data augmentation code only works on the TRAINING set
dst_dir = os.path.join(split_dir, 'Train/')
aug_dir = os.path.join(data_root, 'temp/')
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir, exist_ok=True)
    
    # minimum portion of a bounding box being accepted in a subimage
    overlap_thres = .3
       
    # threshold of non-minor creatures existing in a subimage
    nonminor_thres = .4
    
    # get a list of species that we want to augment (by excluding those we do not want to augment)
    # please change according to your need
    spcs_to_aug = [s for s in bird_spcs_use if s not in ['ROT', 'LAGUA', 'SAT', 'MTRNA', 'OTHRA']]

    # [horizontal filp, vertical flip, left rotate, right rotate, [brightness/contrast tunning, number of images produced]]
    aug_command = [1, 1, 1, 0, [1, 2]]

    dataset_aug(dst_dir, aug_dir, spcs_to_aug, overlap_thres, nonminor_thres, aug_command, 
                img_ext = 'JPEG', annot_file_ext = 'csv', crop_height = 640, crop_width = 640, skip_prob = 0.8)
    
    # move files from aug_dir to dst_dir (train data set)
    for i in glob.glob(os.path.join(aug_dir, '*')):
        shutil.copy2(i, dst_dir) 

#########################################################
## II. Detectron2 dataset register and training models ##
#########################################################

# registering the data in detectron2
img_ext = '.JPEG'
dirs_full = [os.path.join(split_dir, d) for d in os.listdir(split_dir) if d[0].isalnum()]

# Bird species used by object detector. Species contained in dataset that are
# not contained in this list will be categorized as an 'Unknown Bird'

# Bounding box colors for bird species (used when plotting images)
NUM_COLORS = len(bird_spcs_use)
cmap = plt.get_cmap('gist_rainbow')
bird_spcs_use_COLORS = [cmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

register_datasets(
    dirs_full, img_ext, bird_spcs_use, bird_species_colors = bird_spcs_use_COLORS, unknown_bird_category=True
)

# training the multi-species bird detection model
MODEL = 'faster_rcnn'   # {retinanet, faster_rcnn}
model_output_dir = f'./output/Training_models/{MODEL}_{len(bird_spcs_use)}class_set_I_aug_wloss'

if MODEL == 'faster_rcnn' :
    # weight for loss function (unknown class should not be included in dist)
    vid = target_data["Validate"]["class_id"]
    dist = vid.loc[vid.isin(bird_spcs_use)].value_counts()[bird_spcs_use]
    custom_weight = (np.log(max(dist)/dist) + 1).to_list()

    cfg_parms = {'NUM_WORKERS': 0, 'IMS_PER_BATCH': 8, 'BASE_LR': .01, 'GAMMA': 0.001, 'WARMUP_ITERS': 1, 
                 'MAX_ITER': 1500, 'STEPS': [899], 'CHECKPOINT_PERIOD': 899, 'output_dir': model_output_dir,
                 'model_name': "faster_rcnn_R_50_FPN_1x", 'BIRD_SPECIES': bird_spcs_use, 'Custom': True, 
                 'weight': custom_weight}
    
elif MODEL == 'retinanet':
    # TODO: weighted loss not yet tested for retinanet
    custom_weight = []
    cfg_parms = {'NUM_WORKERS': 0, 'IMS_PER_BATCH': 4, 'BASE_LR': .001, 'GAMMA': 0.01, 'WARMUP_ITERS': 1, 
                 'MAX_ITER': 1500, 'STEPS': [899], 'CHECKPOINT_PERIOD': 899, 'output_dir': model_output_dir,
                 'model_name': "retinanet_R_50_FPN_1x", 'BIRD_SPECIES': bird_spcs_use, 'Custom': False}
else:
    raise NotImplementedError
tuned_cfg_params = cfg_parms

# Baysian hyperparameter tunning
tuned_cfg_params = main_hyper(cfg_parms, iterations = 10) 
tuned_cfg_params['MAX_ITER'] = tuned_cfg_params['MAX_ITER'] + 100
tune_weight_dir, loss = main_fit(tuned_cfg_params)
