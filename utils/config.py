# config.py 
#
# functions 
#
# Authors: Krish Kabra, Minxuan Luo, Alexander Xiong, William Lu
# Copyright (C) 2021-2022 Houston Audubon and others

import os 
from detectron2.config import get_cfg
from detectron2 import model_zoo


def add_retinanet_config(args):
    # Create detectron2 config and predictor 
    cfg = get_cfg()
   # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/retinanet_R_50_FPN_1x.yaml")

    # cfg.TEST.EVAL_PERIOD = 20 # set to non-zero integer to get evaluation metric results 
    cfg.DATALOADER.NUM_WORKERS = args.num_workers; 
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = args.learning_rate  
    cfg.SOLVER.MAX_ITER = args.max_iter    
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (bird). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

def add_fasterrcnn_config(args):
    # Create detectron2 config and predictor 
    cfg = get_cfg()
   # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

    # cfg.TEST.EVAL_PERIOD = 20 # set to non-zero integer to get evaluation metric results 
    cfg.DATALOADER.NUM_WORKERS = args.num_workers; 
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = args.learning_rate  
    cfg.SOLVER.MAX_ITER = args.max_iter    
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (bird). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg