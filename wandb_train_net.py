# wandb_train.py
#
# Utilize wandb
#
# Authors: Krish Kabra, Minxuan Luo, Alexander Xiong, William Lu
# Copyright (C) 2021-2022 Houston Audubon and others

import wandb, os
from datetime import datetime
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import default_argument_parser, launch
from detectron2.utils.logger import setup_logger
from utils.config import add_retinanet_config, add_fasterrcnn_config
from utils.dataloader import get_bird_dicts
from utils.trainer import WAndBTrainer

def get_parser():
    parser = default_argument_parser() #  Create a parser with some common arguments used by detectron2 users.
    # directory management
    parser.add_argument('--data_dir',default='./data',type=str, help="path to dataset directory. must contain 'train', 'val', and 'test' folders")
    parser.add_argument('--img_ext',default='.JPEG',type=str, help="image file extension")
    parser.add_argument('--dir_exceptions',default=[],type=list, help="list of folders in dataset directory to be ignored")
    # model
    parser.add_argument('--model_type',default='retinanet',type=str,help='choice of object detector. Options: "retinanet", "faster-rcnn"')
    parser.add_argument('--model_config_file',default="COCO-Detection/retinanet_R_50_FPN_1x.yaml",type=str,help='path to model config file eg. "COCO-Detection/retinanet_R_50_FPN_1x.yaml"')
    parser.add_argument('--pretrained_coco_model_weights',default=True,type=bool,help='load pretrained coco model weights from model config file')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers for dataloader')
    parser.add_argument('--eval_period', default=20, type=int, help='period between coco eval scores on val set')
    parser.add_argument('--max_iter', default=20000, type=int, help='maximum iterations')
    # hyperparams
    parser.add_argument('--learning_rate',default=1e-4,type=float,help='base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--focal_loss_gamma',default=2.0,type=float,help='focal loss gamma (only for retinanet)')
    parser.add_argument('--focal_loss_alpha',default=0.25,type=float,help='focal loss alpha (only for retinanet)')

    parser.add_argument('--output_dir',default='./output/wandb/',type=str,help='output directory for training logs and final model')

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
        DatasetCatalog.register(f"birds_{d}", lambda d=d: get_bird_dicts(os.path.join(data_dir, d), img_ext))
        if f"birds_{d}" in MetadataCatalog.list():
            MetadataCatalog.remove(f"birds_{d}")
        MetadataCatalog.get(f"birds_{d}").set(thing_classes=["bird"])

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
    cfg.DATASETS.TEST = ("birds_val",)  # "birds_test"
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MIN_SIZE_TEST = (640,)

    cfg.OUTPUT_DIR = os.path.join(args.output_dir, f"{args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

def train(cfg):
    # setup training logger
    setup_logger()

    trainer = WAndBTrainer(cfg)
    trainer.resume_or_load(resume=False)

    return trainer.train()

def main(args):
   cfg = setup(args)
   train(cfg)

if __name__ == "__main__":
    args = get_parser().parse_args()

    wandb.init(project='audubon_f21')
    wandb.config.update(args)

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
