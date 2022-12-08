# import some common libraries
import torch
import json
import optuna
import numpy as np
import os
from datetime import datetime

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling import ROI_HEADS_REGISTRY

from utils.custom_loss import CustomFastRCNNOutputLayers
from utils.trainer import MyTrainer


def setup_helper(cfg_parms):
    """
    Helper function for setting-up.
    """   
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{cfg_parms['model_name']}.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{cfg_parms['model_name']}.yaml")

    cfg.DATASETS.TRAIN = ("birds_species_Train",)
    cfg.DATASETS.TEST  = ("birds_species_Validate",)

    cfg.DATALOADER.NUM_WORKERS      = cfg_parms['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH        = cfg_parms['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR              = cfg_parms['BASE_LR']
    cfg.SOLVER.GAMMA                = cfg_parms['GAMMA']
    cfg.SOLVER.WARMUP_ITERS         = cfg_parms['WARMUP_ITERS']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg_parms['BIRD_SPECIES'])
    cfg.SOLVER.MAX_ITER             = cfg_parms['MAX_ITER']
    cfg.SOLVER.STEPS                = cfg_parms['STEPS']
    cfg.SOLVER.CHECKPOINT_PERIOD    = cfg_parms['CHECKPOINT_PERIOD']

    # naming 
    if 'mod_dirname' in cfg_parms:
        mod_dirname = cfg_parms['mod_dirname']
    else:
        mod_dirname = f"{cfg_parms['model_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cfg.OUTPUT_DIR = os.path.join(cfg_parms['output_dir'], mod_dirname)
    
    # save the parameters files
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        with open(os.path.join(cfg.OUTPUT_DIR, 'parameters.txt'), 'a+') as f:
            f.write(json.dumps(cfg_parms))
    return cfg


def setup_default(cfg_parms):
    """
    Set up the model for running. This is the most basic model
    """
    cfg = setup_helper(cfg_parms)
    
    # setup training logger
    setup_logger()
    return cfg


def setup_loss(cfg_parms):
    """
    Set up the model for running the model with custom loss function
    """
    cfg = setup_helper(cfg_parms)

    # Assumes unknown class has no count
    w = cfg_parms['weight'].append(1.0)
    weight = torch.from_numpy(
        np.array(cfg_parms['weight'], dtype='float32')).to("cuda:0")

    # this function makes the custom loss function based off the distribution of the classes
    @ROI_HEADS_REGISTRY.register()
    class CustomROIHeads(StandardROIHeads):
        def __init__(self, cfg, input_shape):
            super().__init__(
                cfg, input_shape, box_predictor = CustomFastRCNNOutputLayers(cfg, input_shape, weight)
            )
    cfg.MODEL.ROI_HEADS.NAME = 'CustomROIHeads'

    # setup training logger
    setup_logger()
    return cfg


# setting up the training
def train(cfg):
    cfg.DATASETS.TRAIN = ("birds_species_Train",)
    cfg.DATASETS.TEST = ("birds_species_Validate",) 
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MIN_SIZE_TEST = (640,)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


# building the model for hyperparameter tuning
def build_model_for_hp(params, cfg_parms):
    cfg_parms['GAMMA'] = params['gamma']
    cfg_parms['BASE_LR'] = params['base_learning_rate']
    cfg = setup_default(cfg_parms)
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
def hp_tune(niter):
    # create study for hyper paramter tuning
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=niter)
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
        cfg = setup_default(cfg_parms)

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