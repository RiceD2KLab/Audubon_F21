import os, sys, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import cv2
from tqdm.autonotebook import tqdm
import torch

#######################################################################################################################
# loading the dataset from a local drive and cropping the images into 640x640 images

from utils.cropping_hank import crop_dataset, crop_dataset_trainer
# #
# # # data_dir is the path that contains both images and annotations (image: jpg; annotation: csv or bbx)
# original_dir = 'C://Users\\VelocityUser\\Documents\\0224_data' # data directory folder
# original_dir = 'C:/Users/VelocityUser/Documents/Audubon_F21/augment_0224_2/'
# # # output dir is the path where you want to output new files. Please use the folder you defined above.
# #
#
#
# data_dir = os.getcwd()+'/0224_aug_3'
#
# #
# os.makedirs(data_dir,exist_ok=True)
# os.makedirs(data_dir+'/titled', exist_ok= True)
# # #
# output_dir = data_dir+'/titled'
#
#
#
# crop_dataset_trainer(original_dir, output_dir, annot_file_ext = 'bbx', crop_height = 640, crop_width = 640,
#                      sliding_size_x=550, sliding_size_y=550, compute_sliding_size=False)
#######################################################################################################################

from utils.cropping_hank import train_val_test_split

# #
# # # create a new output folder for train, val, test dataset
# # # create three folders under the new output folder, with name 'train', 'val', 'test'
#
# os.makedirs(data_dir+'/split', exist_ok= True)
# os.makedirs(data_dir+'/split/train', exist_ok= True)
# os.makedirs(data_dir+'/split/test', exist_ok= True)
# os.makedirs(data_dir+'/split/val', exist_ok= True)
# # #
# crop_dir = data_dir+'/split'
# # # train is a percentage, the fraction of files for training
# train_frac = 0.7
# # the fraction for test is default to be 1-train-val
# val_frac = 0.1


# train_val_test_split(output_dir, crop_dir, train_frac=train_frac,val_frac = val_frac)

# already cropped directory (5 classes)
# crop_dir = 'C://Users\\VelocityUser\\Documents\\D2K TDS A\\Training Tiles'


# already cropped directory (8 classes)
crop_dir = 'C://Users\VelocityUser\Documents\D2K TDS C\TDS C-01'

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
#
# #######################################################################################################################
# # # ************** the data augmenation part in this section of the code *************************************
# # # this data augmentation code only works on the training set!!!
# # # the output direction is "aug_dir"
# #
# import shutil
# from utils.augmentation import AugTrainingSet, dataset_aug
#
# # # dst_dir is the folder of training data(only after cropping)
# dst_dir = crop_dir + '/Train'
# # aug_dir is where we put image after doing data augmentation
# os.makedirs('C://Users\\VelocityUser\\Documents\\Audubon_F21\\temp', exist_ok=True)
# aug_dir = 'C://Users\\VelocityUser\\Documents\\Audubon_F21\\temp'
# #
# #
# # # Minimum portion of a bounding box being accepted in a subimage
# overlap = 0.2
# #
# # # List of species that we want to augment (PLEASE include the full name)
# minor_species = ["Reddish Egret Adult", "White Ibis Adult", "Roseate Spoonbill Adult", "Great Blue Heron Adult", "Great Egret Chick",
#                  "Fly"]
# #
# # # Threshold of non-minor creatures existing in a subimage
# thres = .4
# #
#
# AugTrainingSet(dst_dir, aug_dir, minor_species, overlap, thres, img_ext='JPG', annot_file_ext='csv')
# #
# #
# """"""
# aug_list = glob.glob(os.path.join(aug_dir, '*flipped*'))
# print(aug_list)
# # #
# print(dst_dir)
#
# for i in aug_list:
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
# BIRD_SPECIES = ["Brown Pelican", "Laughing Gull", "Mixed Tern", "Tricolored Heron", "Black Skimmer",
#                 'Black-Crowned Night Heron', 'Reddish Egret', 'White Ibis', 'Roseate Spoonbill',
#                 'Great Blue Heron']
#
# birds_species_names = BIRD_SPECIES
#
# SPECIES_MAP = {0: 'Brown Pelican', 1: 'Laughing Gull', 2: 'Mixed Tern',
#                3: 'Tricolored Heron', 4: 'Black Skimmer', 5: 'Black Crowed Night Heron',
#                6: 'Reddish Egret', 7: 'White Ibis', 8: 'Roseate Spoonbill', 9: 'Great Blue Heron'}


BIRD_SPECIES = ["Brown Pelican Adult", "Laughing Gull Adult", "Great Egret Chick", "Tricolored Heron Adult", "Black Skimmer Adult",
                'Black-Crowned Night Heron Adult', 'Reddish Egret Adult', 'White Ibis Adult', 'Roseate Spoonbill Adult',
                'Great Blue Heron Adult', "Fly", "Brown Pelican Chick ", "Cattle Egret Adult", "Great Egret Adult"]

birds_species_names = BIRD_SPECIES

SPECIES_MAP = {0: 'Brown Pelican Adult', 1: 'Laughing Gull Adult', 2: "Great Egret Chick",
               3: 'Tricolored Heron Adult', 4: 'Black Skimmer Adult', 5: 'Black-Crowned Night Heron Adult',
               6: 'Reddish Egret Adult', 7: 'White Ibis Adult', 8: 'Roseate Spoonbill Adult',
               9: 'Great Blue Heron Adult', 10: 'Fly', 11: "Brown Pelican Chick", 12: "Cattle Egret Adult ",
               13: "Great Egret Adult"}

# #
# # # Bounding box colors for bird species (used when plotting images)
BIRD_SPECIES_COLORS = [(255, 0, 0), (255, 153, 51), (0, 255, 0),
                       (0, 0, 255), (255, 51, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]
#
register_datasets(dirs, img_ext, BIRD_SPECIES, bird_species_colors=BIRD_SPECIES_COLORS, unknown_bird_category=False)
# # #
# # # #
# # # # #######################################################################################################################
# # # #
# # # #
# # # # ########################################################################################################################
# # # # training the bird species model using Faster R-CNN
torch.cuda.empty_cache()

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from utils.trainer import Trainer, MyTrainer
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads import StandardROIHeads

# from utils.custom_loss import CustomFastRCNNOutputLayers

# model_name = "faster_rcnn_R_50_FPN_1x"

# # Create detectron2 config
# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}.yaml"))
#
# # Get pretrained model from MS COCO
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}.yaml")
#
# # add datasets used for training and validation
# cfg.DATASETS.TRAIN = ("birds_species_Train",)
# cfg.DATASETS.TEST = ("birds_species_Validate",)

# # hyperparameters in the cls loss and bbox loss () ## lambd
from detectron2.utils.logger import setup_logger

model_output_dir = '../Training_models/04_07_bay_tune_15class_aug_B'

cfg_parms = {'NUM_WORKERS': 0, 'IMS_PER_BATCH': 6, 'BASE_LR': .001, 'GAMMA': 0.01,
             'WARMUP_ITERS': 1, 'MAX_ITER': 600,
             'STEPS': [499], 'CHECKPOINT_PERIOD': 499, 'output_dir': model_output_dir,
             'model_name': "faster_rcnn_R_50_FPN_1x", 'BIRD_SPECIES': BIRD_SPECIES}

from utils.hyperparameter import main_hyper, main_fit

# hyperparameter tunning
# tuned_cfg_params = main_hyper(cfg_parms, iterations=40)
# tuned_cfg_params['MAX_ITER'] = tuned_cfg_params['MAX_ITER'] + 100
# tune_weight_dir, loss = main_fit(tuned_cfg_params)

############################### hand tuning
# LR_space = np.logspace(-4, -1, 20)
#
# val_loss = []
# dir_name = []
# for i in LR_space:
#     print(i)
#     cfg_parms = {'NUM_WORKERS': 0, 'IMS_PER_BATCH': 6, 'BASE_LR': float(i), 'GAMMA': 0.01,
#                  'WARMUP_ITERS': 1, 'MAX_ITER': 50,
#                  'STEPS': [29], 'CHECKPOINT_PERIOD': 29, 'output_dir': model_output_dir,
#                  'model_name': "faster_rcnn_R_50_FPN_1x", 'BIRD_SPECIES': BIRD_SPECIES}
#
#     fit_dir_parms, loss = main_fit(cfg_parms)
#     val_loss.append(loss)
#     dir_name.append(fit_dir_parms)
#
# val_loss = np.array(val_loss)
#
# indx = np.argmin(val_loss)
#
# tune_weight_dir = dir_name[indx]
# #
# # #
# # ##########################################################################################################################
# # # # # making the output file with using non maximum suppression
# # # #
from utils.cropping_hank import crop_dataset_img_only

#
# # perform tiling on images
data_dir = 'C://Users\\VelocityUser\Documents\\D2K TDS C\\QC C Files'  # data directory folder
os.makedirs(os.getcwd() + '/AI_QC_test/crop', exist_ok=True)
output_dir = os.getcwd() + '/AI_QC_test/crop'
img_ext = '.JPG'
CROP_WIDTH = 640
CROP_HEIGHT = 640
SLIDING_SIZE = 400
# crop_dataset_img_only(data_dir, img_ext, output_dir, crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH,
#                       sliding_size=SLIDING_SIZE)
# #
# # #########################################################################################################################
# #Evaluating the trained detectron2 model
#
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from utils.evaluation import evaluate_full_pipeline

# # create list of tiled images to be run predictor on
eval_file_lst = []
eval_file_lst = eval_file_lst + glob.glob(os.path.join(output_dir, '*.JPEG'))

# Create detectron2 config and predictor
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# location of the trained weights
# cfg.MODEL.WEIGHTS = f"./D2K_TDS_A_5_classes/multibirds_{model_name}/model_final.pth"
# cfg.MODEL.WEIGHTS = tune_weight_dir + '/model_final.pth'

cfg.MODEL.WEIGHTS = 'C://Users\\VelocityUser\\Documents\\Training_models\\04_07_bay_tune_15class_aug_B\\faster_rcnn_R_50_FPN_1x-20220407-232234\\model_final.pth'

# cfg.MODEL.WEIGHTS = 'C://Users\\VelocityUser\\Documents\\Training_models\\04_07_bay_tune_15class_aug_B\\faster_rcnn_R_50_FPN_1x-20220408-050645\\model_final.pth'



# os.getcwd()+'/03_24_bay_tune_8class/faster_rcnn_R_50_FPN_1x-20220325-170316' +'/model_final.pth'

cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.ROI_HEADS.NUM_CLASSES= len(BIRD_SPECIES)

# Create default predictor to run inference
predictor = DefaultPredictor(cfg)
RAW_IMG_WIDTH = 8192
RAW_IMG_HEIGHT = 5460
CROP_WIDTH = 640
CROP_HEIGHT = 640
SLIDING_SIZE = 400
# # #
# # # # Run evaluation
# output_df = evaluate_full_pipeline(eval_file_lst, predictor, SPECIES_MAP, RAW_IMG_WIDTH, RAW_IMG_HEIGHT, CROP_WIDTH,
#                                    CROP_HEIGHT, SLIDING_SIZE)
# output_df.to_csv('D2K_TDS_15_species_hp.csv')
#
# # #######################################################################################################################
# #confusion matrix output for model evaluation

from utils.dataloader import get_bird_species_dicts
from detectron2.data import DatasetCatalog

from utils.confusion_matrix_birds import confusion_matrix_report

# registering the test dataset
d = 1

train_set_dir = crop_dir + '\Test'
DatasetCatalog.register('test_set', lambda d=d: get_bird_species_dicts(train_set_dir, birds_species_names,
                                                                       img_ext='.JPG', unknown_bird_category=True))

data = DatasetCatalog.get("test_set")

# grab the confusion matrix
pred_total, truth_total = confusion_matrix_report(data, predictor, birds_species_names, img_ext='JPG')
#
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(truth_total, pred_total))
print(classification_report(truth_total, pred_total))
