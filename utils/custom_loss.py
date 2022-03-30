import torch
from torch import nn
from detectron2.modeling import FastRCNNOutputLayers, StandardROIHeads
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy

import numpy as np


# this function makes the custom loss function based off the distribution of the classes
@ROI_HEADS_REGISTRY.register()
class MyStandardROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape,
                         box_predictor=MyRCNNOutput(cfg, box_head.output_shape))

# class MyRCNNOutput(FastRCNNOutputLayers):

#     MyRCNNOutput




class weight_loss_head(FastRCNNOutputLayers):
    def __init__(self,  *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # w = num_class.detach().cpu().numpy()
        w = np.array([1000, 200, 200, 200, 200, 100]) # set weight
        w = w/np.sum(w) # uniform weight
        we = torch.Tensor(w)

        self.loss_weight = {"loss_cls": we,"loss_box_reg": loss_weight}

        raise ValueError("custom uniform loss weight")

