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
import cv2
import os, random, ast
from datetime import datetime

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, default_argument_parser, launch
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# import project utility functions 
from utils.config import add_retinanet_config, add_fasterrcnn_config
from utils.dataloader import get_bird_only_dicts, get_bird_species_dicts, register_datasets
from utils.trainer import Trainer
from utils.evaluation import PrecisionRecallEvaluator, plot_precision_recall

BIRD_SPECIES = ["Brown Pelican", "Laughing Gull", "Mixed Tern",
                "Great Blue Heron", "Great Egret/White Morph"]

BIRD_SPECIES_COLORS = [(255, 0, 0), (255, 153, 51), (0, 255, 0),
                       (0, 0, 255), (255, 51, 255)]


def get_parser():
    parser = default_argument_parser()  # Create a parser with some common arguments used by detectron2 users.
    # directory management
    parser.add_argument('--data_dir', default='./data', type=str,
                        help="path to dataset directory. must contain 'train', 'val', and 'test' folders")
    parser.add_argument('--img_ext', default='.JPEG', type=str, help="image file extension")
    parser.add_argument('--dir_exceptions', default=[], type=list,
                        help="list of folders in dataset directory to be ignored")
    # model
    parser.add_argument('--model_type', default='faster-rcnn', type=str,
                        help='choice of object detector. Options: "retinanet", "faster-rcnn"')
    parser.add_argument('--model_config_file', default="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", type=str,
                        help='path to model config file eg. "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"')
    parser.add_argument('--pretrained_weights_file', default="", type=str, help='load pretrained model weights from file. ')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for dataloader')
    parser.add_argument('--eval_period', default=0, type=int, help='period between coco eval scores on val set')
    parser.add_argument('--max_iter', default=3000, type=int, help='maximum epochs')
    parser.add_argument('--checkpoint_period',default=1000,type=int, help='save a checkpoint after this number of iterations')
    # hyperparams
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='base learning rate')
    parser.add_argument('--solver_warmup_factor', type=float, default=0.001, help='warmup factor used for warmup stage of scheduler')
    parser.add_argument('--solver_warmup_iters', type=int, default=100, help='iterations for warmup stage of scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,help='gamma decay factor used in lr scheduler')
    parser.add_argument('--scheduler_steps', default=[1500], help='list/tuple containing lr scheduler iteration steps eg. 1000,2000')
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
    dirs = [os.path.join(data_dir,d) for d in os.listdir(data_dir)
            if d not in dir_exceptions]
    register_datasets(dirs, img_ext, BIRD_SPECIES, bird_species_colors=BIRD_SPECIES_COLORS)

    for d in dirs:
        dataset_dicts = DatasetCatalog.get(f"birds_species_{os.path.basename(d)}")
        for i, k in enumerate(random.sample(dataset_dicts, 3)):
            d = os.path.basename(d)
            img = cv2.imread(k["file_name"])
            visualizer = Visualizer(img[:, :, ::-1],
                                    metadata=MetadataCatalog.get(f"birds_species_{os.path.basename(d)}"), scale=0.5,
                                    instance_mode=ColorMode.SEGMENTATION)
            out = visualizer.draw_dataset_dict(k)
            cv2.imshow(f'{d} example {i}', out.get_image()[:, :, ::-1])
            cv2.waitKey(1)

    # Create detectron2 config
    if args.model_type == 'retinanet':
       cfg = add_retinanet_config(args)
       cfg.MODEL.RETINANET.NUM_CLASSES = len(BIRD_SPECIES)
    elif args.model_type == 'faster-rcnn':
       cfg = add_fasterrcnn_config(args)
       cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(BIRD_SPECIES)
    else:
       raise Exception("Invalid model type entered")

    cfg.OUTPUT_DIR = os.path.join(args.output_dir, f"{args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # setup training logger
    setup_logger()

    return cfg


def train(cfg):
    cfg.DATASETS.TRAIN = ("birds_species_train",)
    cfg.DATASETS.TEST = ("birds_species_val",) # "birds_test"
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MIN_SIZE_TEST = (640,)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    return trainer.train()


def eval(cfg, args):
    # load model weights
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    predictor = DefaultPredictor(cfg)

    cfg.DATASETS.TEST = ("birds_species_val","birds_species_test")

    val_evaluator = PrecisionRecallEvaluator("birds_species_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "birds_species_val")
    print('validation inference:')
    val_precisions, val_max_recalls = inference_on_dataset(predictor.model, val_loader, val_evaluator)
    plot_precision_recall(val_precisions, val_max_recalls, BIRD_SPECIES + ["Unknown Bird"],
                          BIRD_SPECIES_COLORS + [(0, 0, 0)])

    test_evaluator = PrecisionRecallEvaluator("birds_species_test", output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "birds_species_test")
    print('test inference:')
    test_precisions, test_max_recalls = inference_on_dataset(predictor.model, test_loader, test_evaluator)
    plot_precision_recall(test_precisions, test_max_recalls, BIRD_SPECIES + ["Unknown Bird"],
                          BIRD_SPECIES_COLORS + [(0, 0, 0)])

    for d in ["val", "test"]:
        dataset_dicts = DatasetCatalog.get(f"birds_species_{d}")
        print(f'\n {d} examples:')
        for k in random.sample(dataset_dicts, 3):
            im = cv2.imread(k["file_name"])
            outputs = predictor(im)
            outputs = outputs["instances"].to("cpu")
            outputs = outputs[outputs.scores > 0.5]
            v = Visualizer(im[:, :, ::-1],
                           metadata=MetadataCatalog.get(f"birds_species_{d}"),
                           scale=0.5,
                           instance_mode=ColorMode.SEGMENTATION)
            out = v.draw_instance_predictions(outputs)
            cv2.imshow(f'{d} prediction {i}',out.get_image()[:, :, ::-1])
            cv2.waitKey(1)


def main(args):
    cfg = setup(args)
    train(cfg)
    eval(cfg, args)
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
