# config.py 
#
# Authors: Krish Kabra, Minxuan Luo, Alexander Xiong, William Lu
# Copyright (C) 2021-2022 Houston Audubon and others

from detectron2.config import get_cfg
from detectron2 import model_zoo

def add_retinanet_config(args):
    # Create detectron2 config and predictor
    cfg = get_cfg()
    if args.model_config_file != "":
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(args.model_config_file))
        if args.pretrained_coco_model_weights:
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_config_file)

    # Loss parameters
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = args.focal_loss_gamma
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = args.focal_loss_alpha
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    # solver parameters
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.WARMUP_FACTOR = args.solver_warmup_factor
    cfg.SOLVER.WARMUP_ITERS = args.solver_warmup_iters
    cfg.SOLVER.GAMMA = args.scheduler_gamma
    cfg.SOLVER.STEPS = args.scheduler_steps
    cfg.SOLVER.MAX_ITER = args.max_iter
    # other
    cfg.TEST.EVAL_PERIOD = args.eval_period # set to non-zero integer to get evaluation metric results
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    return cfg

def add_fasterrcnn_config(args):
    # Create detectron2 config and predictor 
    cfg = get_cfg()
    if args.model_config_file != "":
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(args.model_config_file))
        if args.pretrained_coco_model_weights:
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_config_file)

    # loss parameters
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    # solver parameters
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.WARMUP_FACTOR = args.solver_warmup_factor
    cfg.SOLVER.WARMUP_ITERS = args.solver_warmup_iters
    cfg.SOLVER.GAMMA = args.scheduler_gamma
    cfg.SOLVER.STEPS = args.scheduler_steps
    cfg.SOLVER.MAX_ITER = args.max_iter
    # other
    cfg.TEST.EVAL_PERIOD = args.eval_period  # set to non-zero integer to get evaluation metric results
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.STEPS = []

    return cfg

