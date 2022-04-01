# train_net.py

from detectron2.utils.logger import setup_logger

import os
from datetime import datetime

from detectron2.engine import DefaultPredictor, default_argument_parser, launch

from detectron2.config import get_cfg
from FasterRCNNDenseNet.densenet import add_densenet_config

from utils.dataloader import register_datasets
from utils.trainer import Trainer
from utils.evaluation import get_precisions_recalls, plot_precision_recall # , PrecisionRecallEvaluator


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
    parser.add_argument('--model_type', default='densenet121', type=str,
                        help='choice of object detector. Options: "densenet121"')
    parser.add_argument('--model_config_file', default="FasterRCNN-DenseNet121.yaml", type=str,
                        help='path to model config file eg. "configs/FasterRCNN-DenseNet121.yaml"')
    parser.add_argument('--pretrained_weights_file', default="", type=str, help='load pretrained model weights from file. ')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for dataloader')
    parser.add_argument('--eval_period', default=0, type=int, help='period between coco eval scores on val set')
    parser.add_argument('--max_iter', default=3000, type=int, help='maximum epochs')
    parser.add_argument('--checkpoint_period',default=1000,type=int, help='save a checkpoint after this number of iterations')
    # hyperparameters
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='base learning rate')
    parser.add_argument('--solver_warmup_factor', type=float, default=0.001, help='warmup factor used for warmup stage of scheduler')
    parser.add_argument('--solver_warmup_iters', type=int, default=100, help='iterations for warmup stage of scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,help='gamma decay factor used in lr scheduler')
    parser.add_argument('--scheduler_steps', default=[1500], help='list/tuple containing lr scheduler iteration steps eg. 1000,2000')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
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

    cfg = get_cfg()
    add_densenet_config(cfg, args)
    # cfg.merge_from_file(args.config_file)   # done inside add_densenet_function
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.DENSENET.NUM_CLASSES = len(BIRD_SPECIES)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(BIRD_SPECIES)
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # 512 default
    # cfg.freeze()   # what does this do?
    # default_setup(cfg, args)   # doubt I need this

    # TODO: Keep this output directory name?
    cfg.OUTPUT_DIR = os.path.join(args.output_dir, f"{args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # setup training logger
    setup_logger()

    return cfg


def train(cfg):
    cfg.DATASETS.TRAIN = ("birds_species_train",)
    cfg.DATASETS.TEST = ("birds_species_val",)   # "birds_test"
    cfg.INPUT.MIN_SIZE_TRAIN = 640   # [640,]
    cfg.INPUT.MIN_SIZE_TEST = 640

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)   # add args.resume to arguments?

    return trainer.train()


def eval(cfg):
    # removed args from function arguments b/c not used
    # load model weights
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    predictor = DefaultPredictor(cfg)

    cfg.DATASETS.TEST = ("birds_species_val", "birds_species_test")

    print('validation inference:')
    val_precisions, val_max_recalls = get_precisions_recalls(cfg, predictor, "birds_species_val")
    plot_precision_recall(val_precisions, val_max_recalls, BIRD_SPECIES + ["Unknown Bird"],
                          BIRD_SPECIES_COLORS + [(0, 0, 0)])

    print('test inference:')
    test_precisions, test_max_recalls = get_precisions_recalls(cfg, predictor, "birds_species_test")
    plot_precision_recall(test_precisions, test_max_recalls, BIRD_SPECIES + ["Unknown Bird"],
                          BIRD_SPECIES_COLORS + [(0, 0, 0)])

    # for d in ["val", "test"]:
    #     dataset_dicts = DatasetCatalog.get(f"birds_species_{d}")
    #     print(f'\n {d} examples:')
    #     for i,k in enumerate(random.sample(dataset_dicts, 3)):
    #         im = cv2.imread(k["file_name"])
    #         outputs = predictor(im)
    #         outputs = outputs["instances"].to("cpu")
    #         outputs = outputs[outputs.scores > 0.5]
    #         v = Visualizer(im[:, :, ::-1],
    #                        metadata=MetadataCatalog.get(f"birds_species_{d}"),
    #                        scale=0.5,
    #                        instance_mode=ColorMode.SEGMENTATION)
    #         out = v.draw_instance_predictions(outputs)
    #         cv2.imshow(f'{d} prediction {i}',out.get_image()[:, :, ::-1])
    #         cv2.waitKey(1)


def main(args):
    cfg = setup(args)
    train(cfg)
    eval(cfg)
    # cv2.waitkey(0)
    # print("Press any key to continue...")
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    
    args = get_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,   # does this exist?
        args=(args,)
    )