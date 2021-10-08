# train_net.py 
#
# Train object detection model on Houston Audubon dataset
#
# Authors: Krish Kabra, Minxuan Luo, Alexander Xiong, William Lu
# Copyright (C) 2021-2022 Houston Audubon and others
#
# WORK IN PROGRESS:
# 1. Update to include loading model weights from checkpoint 
# 2. Setup evaluation only mode 

import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random
import glob, os
import matplotlib.pyplot as plt
from skimage import io 
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, launch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# import project utility functions 
from utils.dataloader import get_bird_dicts
from utils.trainer import Trainer 


def get_parser():
    parser = default_argument_parser() #  Create a parser with some common arguments used by detectron2 users.
    parser.add_argument('--data_dir',type=str, help="path to dataset directory")
    parser.add_argument('--img_ext',default='.JPG',type=str, help="image file extension")
    parser.add_argument('--dir_exceptions',type=list, help="list of folders in dataset directory to be ignored")
   
    parse.add_argument('--model_type',default='retinanet',type=str,help='choice of object detector. Options: "retinanet"')
    parse.add_argument('--num_workers',default=2,type=int,help='number of workers for dataloader')
    parse.add_argument('--learning_rate',default=0.0001,,help='base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
    parse.add_argument('--max_iter',default=1e-4,type=float,help='maximum epochs')
    parse.add_argument('--batch_size',default=64,type=int,help='batch size')

    parse.add_argument('--output_dir',default='./output',type=str,help='output directory for training logs and final model')
    # parse.add_argument('--',default=,type=,help=)

    return parser

def setup(args): 
    # data setup
    data_dir = args.data_dir
    img_ext = args.img_ext
    dir_exceptions = args.dir_exceptions 

    dirs = [d for d in os.listdir(data_dir) 
              if d not in dir_exceptions]
    for d in dirs: 
      if f"birds_{d}" in DatasetCatalog.list():
          DatasetCatalog.remove(f"birds_{d}")
      DatasetCatalog.register(f"birds_{d}", lambda d=d: get_bird_dicts(os.path.join(data_dir,d),img_ext))
      if f"birds_{d}" in MetadataCatalog.list():
          MetadataCatalog.remove(f"birds_{d}")
      MetadataCatalog.get(f"birds_{d}").set(thing_classes=["bird"])

      #dataset_dicts = get_bird_dicts(os.path.join(data_dir,d),img_ext)
      #print(f"\n {d} examples:")
      #for k in random.sample(dataset_dicts, 3):
      #    img = cv2.imread(k["file_name"])
      #    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(f"birds_{d}"), scale=0.5)
      #    out = visualizer.draw_dataset_dict(k)
      #    cv2_imshow(out.get_image()[:, :, ::-1])

    # Create detectron2 config 
    if args.model_type == 'retinanet': 
        cfg = add_retinanet_config(args)
    else: 
        raise Exception("Invalid model type entered")

    cfg.DATASETS.TRAIN = ("birds_train",)
    cfg.DATASETS.TEST = ("birds_val",) # "birds_test"
            
    return cfg


def train(cfg): 
    # setup training logger
    setup_logger()
       
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)

    return trainer.train() 

def eval(cfg): 
    # load model weights
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    predictor = DefaultPredictor(cfg)
    
    val_evaluator = COCOEvaluator("birds_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "birds_val")
    print('validation inference:',inference_on_dataset(predictor.model, val_loader, val_evaluator))
    test_evaluator = COCOEvaluator("birds_test", output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "birds_test")
    print('test inference:',inference_on_dataset(predictor.model, test_loader, test_evaluator))

    #for d in ["val", "test"]: 
    #    dataset_dicts = get_bird_dicts(os.path.join(data_dir,d),img_ext)
    #    print(f'\n {d} examples:')
    #    for k in random.sample(dataset_dicts, 3):    
    #        im = cv2.imread(k["file_name"])
    #        outputs = predictor(im)  
    #        outputs = outputs["instances"].to("cpu")
    #        outputs = outputs[outputs.scores > 0.8]
    #        v = Visualizer(im[:, :, ::-1],
    #                    metadata=MetadataCatalog.get(f"birds_{d}"), 
    #                    scale=0.5, 
    #                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #        )
    #        out = v.draw_instance_predictions(outputs)
    #        cv2_imshow(out.get_image()[:, :, ::-1])
    #        cv2.waitKey(1)

def main(args): 
    cfg = setup(args)
    train(cfg, args)
    eval(cfg)
    #cv2.destroyAllWindows()
    pass  

if __name__ == "__main__":
    
    args = get_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )