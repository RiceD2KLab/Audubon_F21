
from detectron2.config import CfgNode as CN
# From Audubon
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# import os

def add_densenet_config(cfg, args):
    """
    Add config for DenseNet.
    Needs to be called before cfg.merge_from_file so that DENSENET exits.
    """

    # model_channels = {
    # 'densenet121': 1024,
    # 'densenet161': 2208,
    # 'densenet169': 1664,
    # 'densenet201': 1920,
    # }

    _C = cfg

    # This will allow us to add custom attributes, which can be overwritten
    _C.MODEL.DENSENET = CN()

    # These are all defined in a config file
    # _C.MODEL.DENSENET.OUT_FEATURES = ["SoleStage"]

    # _C.MODEL.DENSENET.CONV_BODY = "densenet121"
    # _C.MODEL.DENSENET.OUT_CHANNELS = model_channels["densenet121"]
    # _C.MODEL.DENSENET.PRETRAINED = True

    ####################### ADAPTED FROM AUDUBON_F21 ########################
    if args.model_config_file != "":
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(args.model_config_file)
        if args.pretrained_weights_file != "":
            cfg.MODEL.WEIGHTS = args.pretrained_weights_file

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
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period  # set to non-zero integer to get evaluation metric results
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    return cfg

