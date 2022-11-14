from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import cv2
import os, random, ast
from datetime import datetime

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, default_argument_parser, launch
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
# from detectron2.evaluation import COCOEvaluator, innference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import CfgNode as CN

from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling import ROI_HEADS_REGISTRY

#

# import project utility functions on PC
# from utils.custom_loss import CustomFastRCNNOutputLayers
# from utils.config import add_retinanet_config, add_fasterrcnn_config
# from utils.dataloader import get_bird_only_dicts, get_bird_species_dicts, register_datasets
# from utils.trainer import Trainer, MyTrainer
# from utils.evaluation import PrecisionRecallEvaluator, get_precisions_recalls, plot_precision_recall

# For google colab
from utils.custom_loss import CustomFastRCNNOutputLayers
from utils.config import add_retinanet_config, add_fasterrcnn_config
from utils.dataloader import get_bird_only_dicts, get_bird_species_dicts, register_datasets
from utils.trainer import Trainer, MyTrainer
from utils.evaluation import PrecisionRecallEvaluator, get_precisions_recalls, plot_precision_recall


import json
import optuna

# Set up the model for running. This is the most basic model
def setup(cfg_parms):
    # outputs the best hyperparameter tuned model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{cfg_parms['model_name']}.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{cfg_parms['model_name']}.yaml")

    cfg.DATASETS.TRAIN = ("birds_species_train",)
    cfg.DATASETS.TEST = ("birds_species_val",)

    cfg.DATALOADER.NUM_WORKERS = cfg_parms['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH = cfg_parms['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = cfg_parms['BASE_LR']
    cfg.SOLVER.GAMMA = cfg_parms['GAMMA']
    cfg.SOLVER.WARMUP_ITERS = cfg_parms['WARMUP_ITERS']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg_parms['BIRD_SPECIES'])
    cfg.SOLVER.MAX_ITER = cfg_parms['MAX_ITER']
    cfg.SOLVER.STEPS = cfg_parms['STEPS']
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg_parms['CHECKPOINT_PERIOD']
    cfg.MODEL.weight = cfg_parms['weight']

    # naming needs to be updated
    cfg.OUTPUT_DIR = os.path.join(cfg_parms['output_dir'],
                                  f"{cfg_parms['model_name']}-{'best_tune'}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # save the parameters files
    with open(os.path.join(cfg.OUTPUT_DIR, 'parameters.txt'), 'a+') as f:
        f.write(json.dumps(cfg_parms))

    # setup training logger
    setup_logger()

    return cfg

# Set up the model for running. This is the most basic model

def setup1(cfg_parms):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{cfg_parms['model_name']}.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{cfg_parms['model_name']}.yaml")

    cfg.DATASETS.TRAIN = ("birds_species_Train",)
    cfg.DATASETS.TEST = ("birds_species_Validate",)

    cfg.DATALOADER.NUM_WORKERS = cfg_parms['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH = cfg_parms['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = cfg_parms['BASE_LR']
    cfg.SOLVER.GAMMA = cfg_parms['GAMMA']
    cfg.SOLVER.WARMUP_ITERS = cfg_parms['WARMUP_ITERS']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg_parms['BIRD_SPECIES'])
    cfg.SOLVER.MAX_ITER = cfg_parms['MAX_ITER']
    cfg.SOLVER.STEPS = cfg_parms['STEPS']
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg_parms['CHECKPOINT_PERIOD']

    # naming needs to be updated
    cfg.OUTPUT_DIR = os.path.join(cfg_parms['output_dir'],
                                  f"{cfg_parms['model_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)



    # save the parameters files
    with open(os.path.join(cfg.OUTPUT_DIR, 'parameters.txt'), 'a+') as f:
        f.write(json.dumps(cfg_parms))

    # setup training logger
    setup_logger()

    return cfg

# Set up the model for running the model with custom loss function
import torch

def setup_loss(cfg_parms):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{cfg_parms['model_name']}.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{cfg_parms['model_name']}.yaml")

    cfg.DATASETS.TRAIN = ("birds_species_Train",)
    cfg.DATASETS.TEST = ("birds_species_Validate",)

    cfg.DATALOADER.NUM_WORKERS = cfg_parms['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH = cfg_parms['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = cfg_parms['BASE_LR']
    cfg.SOLVER.GAMMA = cfg_parms['GAMMA']
    cfg.SOLVER.WARMUP_ITERS = cfg_parms['WARMUP_ITERS']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg_parms['BIRD_SPECIES'])
    cfg.SOLVER.MAX_ITER = cfg_parms['MAX_ITER']
    cfg.SOLVER.STEPS = cfg_parms['STEPS']
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg_parms['CHECKPOINT_PERIOD']

    # Assumes unknown class has no count
    w = cfg_parms['weight'].append(1.0)
    weight = torch.from_numpy(
        np.array(cfg_parms['weight'], dtype='float32')).to("cuda:0")

    # ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
    # this function makes the custom loss function based off the distribution of the classes
    @ROI_HEADS_REGISTRY.register()
    class CustomROIHeads(StandardROIHeads):
        def __init__(self, cfg, input_shape):
            # gpu_weight =
            super().__init__(cfg, input_shape,
                             box_predictor=CustomFastRCNNOutputLayers(cfg, input_shape, weight))

    cfg.MODEL.ROI_HEADS.NAME = 'CustomROIHeads'

    # naming needs to be updated
    cfg.OUTPUT_DIR = os.path.join(cfg_parms['output_dir'],
                                  f"{cfg_parms['model_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)



    # save the parameters files
    with open(os.path.join(cfg.OUTPUT_DIR, 'parameters.txt'), 'a+') as f:
        f.write(json.dumps(cfg_parms))

    # setup training logger
    setup_logger()

    return cfg

# Set up the model for running the Faster RCNN with a dense net backbone
def setup_dense(cfg_parms):
    import FasterRCNNDenseNet.densenet.densenet

    # setup training logger
    setup_logger()

    cfg = get_cfg()
    cfg.MODEL.DENSENET = CN()
    cfg.MODEL.DENSENET.OUT_FEATURES = ["SoleStage"]
    cfg.MODEL.DENSENET.PRETRAINED = True
    cfg.MODEL.DENSENET.CONV_BODY = "densenet121"
    cfg.MODEL.DENSENET.OUT_CHANNELS = 1024

    cfg.merge_from_file(
        "C://Users\\VelocityUser\\Documents\\Audubon_F21\\FasterRCNNDenseNet\\configs\\FasterRCNN-DenseNet121.yaml")



    cfg.DATALOADER.NUM_WORKERS = cfg_parms['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH = cfg_parms['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = cfg_parms['BASE_LR']
    cfg.SOLVER.GAMMA = cfg_parms['GAMMA']
    cfg.SOLVER.WARMUP_ITERS = cfg_parms['WARMUP_ITERS']
    cfg.MODEL.DENSENET.NUM_CLASSES = len(cfg_parms['BIRD_SPECIES'])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg_parms['BIRD_SPECIES'])
    cfg.SOLVER.MAX_ITER = cfg_parms['MAX_ITER']
    cfg.SOLVER.STEPS = cfg_parms['STEPS']
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg_parms['CHECKPOINT_PERIOD']

    cfg.DATASETS.TRAIN = ("birds_species_Train",)
    cfg.DATASETS.TEST = ("birds_species_Validate",)


    # naming needs to be updated
    cfg.OUTPUT_DIR = os.path.join(cfg_parms['output_dir'],
                                  f"{cfg_parms['model_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # save the parameters files
    with open(os.path.join(cfg.OUTPUT_DIR, 'parameters.txt'), 'a+') as f:
        f.write(json.dumps(cfg_parms))



    return cfg

# setting up the training
def train(cfg):
    cfg.DATASETS.TRAIN = ("birds_species_Train",)
    cfg.DATASETS.TEST = ("birds_species_Validate",)  # "birds_test"
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MIN_SIZE_TEST = (640,)

    # trainer = Trainer(cfg)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


# evaluting the model for the hyperparameter tuning. This is performed on the validation st
def eval(cfg, cfg_parms):
    '''
    Args:
        cfg: the model that we are using for the predictor (so faster R-CNN)
        cfg_parms: is a dictionary of all the parameters needed for t

    Returns:
        the average precision for each classes

        the classification loss function for IOU threshold of 50%
    '''

    # load model weights

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    predictor = DefaultPredictor(cfg)

    cfg.DATASETS.TEST = "birds_species_Validate"

    # print('validation inference:')
    val_precisions, val_max_recalls = get_precisions_recalls(cfg, predictor, "birds_species_Validate")

    return sum(val_precisions) / len(val_precisions)


# building the model for hyperparameter tuning
def build_model_for_hp(params, cfg_parms):
    cfg_parms['GAMMA'] = params['gamma']
    cfg_parms['BASE_LR'] = params['base_learning_rate']

    # cfg = setup_dense(cfg_parms)

    cfg = setup1(cfg_parms)

    return cfg

# objective function for HP tuning. 
def objective(trial):
    params = {
        'base_learning_rate': trial.suggest_loguniform('base_learning_rate', 1e-4, 0.1),
        'gamma': trial.suggest_loguniform('gamma', 0.001, 0.2)
    }
    # if learning rate is too high then output a large validation loss
    try:
        model_cfg = build_model_for_hp(params, cfg_parms_cp)
        # train on bird species
        train(model_cfg)

        # find the last validation loss value produced by the model
        metrics = os.path.join(model_cfg.OUTPUT_DIR, 'metrics.json')

        f = open(metrics, 'r')
        
        for i in f:
            data = json.loads(i)
            if data['iteration'] == model_cfg.SOLVER.MAX_ITER - 1:
                val_loss = data['validation_loss']

    except:
        val_loss = 1000

    return val_loss

# bayesian hyperparameter optimization
def hp_tune(iter):
    # create study for hyper paramter tuning
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=iter)
    return study

# pipeline for hyperparameter tuning.
def main_hyper(cfg_parms, iterations):
    global cfg_parms_cp
    cfg_parms_cp = cfg_parms
    study = hp_tune(iterations)
    trial = study.best_trial
    print(trial.value, "when params are", trial.params)

    cfg_parms['BASE_LR'] = trial.params['base_learning_rate']
    cfg_parms['GAMMA'] = trial.params['gamma']

    return cfg_parms


# only one model fit
def main_fit(cfg_parms):
    if cfg_parms["Custom"] == True:
        cfg = setup_loss(cfg_parms)
    else:
        cfg = setup1(cfg_parms)


    train(cfg)

    # find the last validation loss value produced by the model
    metrics = os.path.join(cfg.OUTPUT_DIR, 'metrics.json')

    f = open(metrics, 'r')

    for i in f:
        data = json.loads(i)
        if data['iteration'] == cfg.SOLVER.MAX_ITER - 1:
            val_loss = data['validation_loss']
    print(' val loss is: ', val_loss)

    return cfg.OUTPUT_DIR, val_loss
