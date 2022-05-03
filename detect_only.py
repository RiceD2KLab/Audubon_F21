from utils.cropping_hank import crop_dataset_img_only
import os, sys, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import cv2
from tqdm.autonotebook import tqdm
import torch

def run_default():
    crop_dir = 'C://Users\VelocityUser\Documents\D2K TDS A\TDS A-10'

    BIRD_SPECIES = ['BRPEA', 'LAGUA', 'MTRNA', 'TRHEA', 'BLSKA',
                    'BCNHA', 'REEGA', 'WHIBA', 'ROSPA',
                    'GBHEA']

    SPECIES_MAP = {0: 'BRPEA', 1: 'LAGUA', 2: 'MTRNA',
                   3: 'TRHEA', 4: 'BLSKA', 5: 'BCNHA',
                   6: 'REEGA', 7: 'WHIBA', 8: 'ROSPA', 9: 'GBHEA'}

    birds_species_names = BIRD_SPECIES

    # # perform tiling on images 8K images
    data_dir = 'C://Users\\VelocityUser\Documents\\D2K TDS B\\AI QC B'  # data directory folder
    os.makedirs(os.getcwd() + '/AI_QC_test/crop', exist_ok=True)
    output_dir = os.getcwd() + '/AI_QC_test/crop'
    img_ext = '.JPG'
    CROP_WIDTH = 640
    CROP_HEIGHT = 640
    SLIDING_SIZE = 400
    # crop_dataset_img_only(data_dir, img_ext, output_dir, crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH,
    #                       sliding_size=SLIDING_SIZE)
    #
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model

    # location of the trained weights
    # cfg.MODEL.WEIGHTS = f"./D2K_TDS_A_5_classes/multibirds_{model_name}/model_final.pth"
    # cfg.MODEL.WEIGHTS = tune_weight_dir + '/model_final.pth'

    cfg.MODEL.WEIGHTS = 'C://Users\\VelocityUser\\Documents\\Training_models\\03_24_bay_tune_10class_aug_B\\' \
                        'faster_rcnn_R_50_FPN_1x-20220402-174351\\model_final.pth '

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(BIRD_SPECIES)

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
    # output_df.to_csv('D2K_TDS_10_species_QC_B.csv')
    #
    # # #######################################################################################################################
    # #confusion matrix output for model evaluation

    from utils.dataloader import get_bird_species_dicts
    from detectron2.data import DatasetCatalog
    from sklearn.metrics import confusion_matrix, classification_report
    from utils.confusion_matrix_birds import confusion_matrix_report

    # registering the test dataset
    d = 1

    train_set_dir = crop_dir + '\Test'
    DatasetCatalog.register('test_set', lambda d=d: get_bird_species_dicts(train_set_dir, birds_species_names,
                                                                           img_ext='.JPG', unknown_bird_category=True))
    data = DatasetCatalog.get("test_set")

    # grab the confusion matrix
    pred_total, truth_total = confusion_matrix_report(data, predictor, birds_species_names, img_ext='JPG')

    print(confusion_matrix(truth_total, pred_total))
    print(classification_report(truth_total, pred_total))

    ########################################################################################################################
    # precision and recall curve
    from utils.evaluation import get_precisions_recalls, plot_precision_recall

    cfg.DATASETS.TEST = ("test_set",)

    from utils.dataloader import register_datasets

    data_dir = crop_dir
    img_ext = '.JPG'
    dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]

    BIRD_SPECIES_COLORS = [(255, 0, 0), (255, 153, 51), (0, 255, 0),
                           (0, 0, 255), (255, 51, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                           (255, 255, 255)]
    register_datasets(dirs, img_ext, BIRD_SPECIES, bird_species_colors=BIRD_SPECIES_COLORS, unknown_bird_category=False)

    # change this to the specific directory of the fitting model
    cfg.OUTPUT_DIR = 'C://Users\\VelocityUser\\Documents\\Training_models\\03_24_bay_tune_10class_aug_B\\faster_rcnn_R_50_FPN_1x-20220402-174351'

    # print('validation inference:')
    val_precisions, val_max_recalls = get_precisions_recalls(cfg, predictor, "birds_species_Test")
    plot_precision_recall(val_precisions, val_max_recalls, BIRD_SPECIES + ["Unknown Bird"],
                          BIRD_SPECIES_COLORS + [(0, 0, 0)])


def run(argv):
    crop_dir = 'C://Users\VelocityUser\Documents\detect_images'

    BIRD_SPECIES = ['BRPEA', 'LAGUA', 'MTRNA', 'TRHEA', 'BLSKA',
                    'BCNHA', 'REEGA', 'WHIBA', 'ROSPA',
                    'GBHEA']

    SPECIES_MAP = {0: 'BRPEA', 1: 'LAGUA', 2: 'MTRNA',
                   3: 'TRHEA', 4: 'BLSKA', 5: 'BCNHA',
                   6: 'REEGA', 7: 'WHIBA', 8: 'ROSPA', 9: 'GBHEA'}

    birds_species_names = BIRD_SPECIES

    # # perform tiling on images 8K images
    data_dir = 'C://Users\\VelocityUser\Documents\\detect_images\\AI QC B'  # data directory folder
    os.makedirs(os.getcwd() + '/AI_QC_test/crop', exist_ok=True)
    output_dir = os.getcwd() + '/AI_QC_test/crop'
    img_ext = '.JPG'
    CROP_WIDTH = 640
    CROP_HEIGHT = 640
    SLIDING_SIZE = 400
    # crop_dataset_img_only(data_dir, img_ext, output_dir, crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH,
    #                       sliding_size=SLIDING_SIZE)
    #
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model

    # location of the trained weights
    # cfg.MODEL.WEIGHTS = f"./D2K_TDS_A_5_classes/multibirds_{model_name}/model_final.pth"
    # cfg.MODEL.WEIGHTS = tune_weight_dir + '/model_final.pth'

    cfg.MODEL.WEIGHTS = str(argv[0]) + "\\" + os.listdir(str(argv[0])).sort()[-1] + "\\model_final.pth"

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(BIRD_SPECIES)

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
    # output_df.to_csv('D2K_TDS_10_species_QC_B.csv')
    #
    # # #######################################################################################################################
    # #confusion matrix output for model evaluation

    from utils.dataloader import get_bird_species_dicts
    from detectron2.data import DatasetCatalog
    from sklearn.metrics import confusion_matrix, classification_report
    from utils.confusion_matrix_birds import confusion_matrix_report

    # registering the test dataset
    d = 1

    train_set_dir = crop_dir + '\Test'
    DatasetCatalog.register('test_set', lambda d=d: get_bird_species_dicts(train_set_dir, birds_species_names,
                                                                           img_ext='.JPG', unknown_bird_category=True))
    data = DatasetCatalog.get("test_set")

    # grab the confusion matrix
    pred_total, truth_total = confusion_matrix_report(data, predictor, birds_species_names, img_ext='JPG')

    print(confusion_matrix(truth_total, pred_total))
    print(classification_report(truth_total, pred_total))

    ########################################################################################################################
    # precision and recall curve
    from utils.evaluation import get_precisions_recalls, plot_precision_recall

    cfg.DATASETS.TEST = ("test_set",)

    from utils.dataloader import register_datasets

    data_dir = crop_dir
    img_ext = '.JPG'
    dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]

    BIRD_SPECIES_COLORS = [(255, 0, 0), (255, 153, 51), (0, 255, 0),
                           (0, 0, 255), (255, 51, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                           (255, 255, 255)]
    register_datasets(dirs, img_ext, BIRD_SPECIES, bird_species_colors=BIRD_SPECIES_COLORS, unknown_bird_category=False)

    # change this to the specific directory of the fitting model
    cfg.OUTPUT_DIR = str(argv[0]) + "\\" + os.listdir(str(argv[0])).sort()[-1]

    # print('validation inference:')
    val_precisions, val_max_recalls = get_precisions_recalls(cfg, predictor, "birds_species_Test")
    plot_precision_recall(val_precisions, val_max_recalls, BIRD_SPECIES + ["Unknown Bird"],
                          BIRD_SPECIES_COLORS + [(0, 0, 0)])


if __name__ == "__main__":
    run_default()

