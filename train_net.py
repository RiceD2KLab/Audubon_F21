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
# 3. setup checkpointing 

import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import cv2
import os, random
from datetime import datetime

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, default_argument_parser, launch
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# import project utility functions 
from utils.config import add_retinanet_config, add_fasterrcnn_config
from utils.dataloader import get_bird_dicts
from utils.trainer import Trainer


def get_parser():
    parser = default_argument_parser()  # Create a parser with some common arguments used by detectron2 users.
    # directory management
    parser.add_argument('--data_dir', default='./data', type=str,
                        help="path to dataset directory. must contain 'train', 'val', and 'test' folders")
    parser.add_argument('--img_ext', default='.JPEG', type=str, help="image file extension")
    parser.add_argument('--dir_exceptions', default=[], type=list,
                        help="list of folders in dataset directory to be ignored")
    # model
    parser.add_argument('--model_type', default='retinanet', type=str,
                        help='choice of object detector. Options: "retinanet", "faster-rcnn"')
    parser.add_argument('--model_config_file', default="COCO-Detection/retinanet_R_50_FPN_1x.yaml", type=str,
                        help='path to model config file eg. "COCO-Detection/retinanet_R_50_FPN_1x.yaml"')
    parser.add_argument('--pretrained_coco_model_weights', default=True, type=bool,
                        help='load pretrained coco model weights from model config file')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers for dataloader')
    parser.add_argument('--eval_period', default=0, type=int, help='period between coco eval scores on val set')
    parser.add_argument('--max_iter', default=500, type=int, help='maximum epochs')
    # hyperparams
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--focal_loss_gamma', default=2.0, type=float, help='focal loss gamma (only for retinanet)')
    parser.add_argument('--focal_loss_alpha', default=0.25, type=float, help='focal loss alpha (only for retinanet)')

    parser.add_argument('--output_dir', default='./output', type=str,
                        help='output directory for training logs and final model')

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

     dataset_dicts = get_bird_dicts(os.path.join(data_dir,d),img_ext)
     for i,k in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(k["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(f"birds_{d}"), scale=0.5)
        out = visualizer.draw_dataset_dict(k)
        cv2.imshow(f'{d} example {i}',out.get_image()[:, :, ::-1])
        cv2.waitKey(1)

   # Create detectron2 config
   if args.model_type == 'retinanet':
       cfg = add_retinanet_config(args)
       cfg.MODEL.RETINANET.NUM_CLASSES = 1
   elif args.model_type == 'faster-rcnn':
       cfg = add_fasterrcnn_config(args)
       cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
   else:
       raise Exception("Invalid model type entered")

   cfg.DATASETS.TRAIN = ("birds_train",)
   cfg.DATASETS.TEST = ("birds_val",) # "birds_test"
   cfg.INPUT.MIN_SIZE_TRAIN = (640,)
   cfg.INPUT.MIN_SIZE_TEST = (640,)

   cfg.OUTPUT_DIR = os.path.join(args.output_dir, f"{args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
   os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

            
   return cfg


def train(cfg):
   # setup training logger
   setup_logger()
       
   trainer = Trainer(cfg)
   trainer.resume_or_load(resume=False)

   return trainer.train()

def eval(cfg, args):
   # load model weights
   cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
   predictor = DefaultPredictor(cfg)
    
   cfg.DATASETS.TEST = ("birds_val","birds_test")

   val_evaluator = COCOEvaluator("birds_val", output_dir=cfg.OUTPUT_DIR)
   val_loader = build_detection_test_loader(cfg, "birds_val")
   print('validation inference:',inference_on_dataset(predictor.model, val_loader, val_evaluator))
   test_evaluator = COCOEvaluator("birds_test", output_dir=cfg.OUTPUT_DIR)
   test_loader = build_detection_test_loader(cfg, "birds_test")
   print('test inference:',inference_on_dataset(predictor.model, test_loader, test_evaluator))

   for d in ["val", "test"]:
       dataset_dicts = get_bird_dicts(os.path.join(args.data_dir,d),args.img_ext)
       print(f'\n {d} examples:')
       for k in random.sample(dataset_dicts, 3):
           im = cv2.imread(k["file_name"])
           outputs = predictor(im)
           outputs = outputs["instances"].to("cpu")
           outputs = outputs[outputs.scores > 0.8]
           v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(f"birds_{d}"),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. this option is only available for segmentation models
           )
           out = v.draw_instance_predictions(outputs)
           cv2.imshow(f'{d} prediction {i}',out.get_image()[:, :, ::-1])
           cv2.waitKey(1)

def main(args):
   cfg = setup(args)
   train(cfg)
   eval(cfg,args)
   cv2.waitkey(0)
   print("Press any key to continue...")
   cv2.destroyAllWindows()

if __name__ == "__main__":
    
    args = get_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
