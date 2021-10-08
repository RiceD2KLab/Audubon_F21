import numpy as np
import torch
import time
from datetime import timedelta

from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class ValidationLossHook(HookBase): 
  """
  Code adapted from: https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
  """
  def __init__(self, model, data_loader, eval_period=20):
    self._model = model
    self._period = eval_period
    self._data_loader = data_loader
  
  def _do_loss_eval(self): 
    losses = []
    for idx, inputs in enumerate(self._data_loader):            
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      loss_batch = self._get_loss(inputs)
      losses.append(loss_batch)
    
    mean_loss = np.mean(losses)
    self.trainer.storage.put_scalar('validation_loss', mean_loss)
    comm.synchronize()

    return losses
          
  def _get_loss(self, data):
    # How loss is calculated on train_loop 
    metrics_dict = self._model(data)
    metrics_dict = {
        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in metrics_dict.items()
    }
    total_losses_reduced = sum(loss for loss in metrics_dict.values())
    return total_losses_reduced
    
  def after_step(self):
    next_iter = self.trainer.iter + 1
    is_final = next_iter == self.trainer.max_iter
    if is_final or (self._period > 0 and next_iter % self._period == 0):
      self._do_loss_eval()

# Build custom trainer with validation loss hook + evaluation  


class Trainer(DefaultTrainer): 
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:     
      return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    else: 
      return COCOEvaluator(dataset_name, output_dir=output_folder)

  def build_hooks(self): 
    hooks = super().build_hooks()
    hooks.append(ValidationLossHook(
                              self.model,
                              build_detection_test_loader(
                                self.cfg,
                                self.cfg.DATASETS.TEST[0], # assume first testing dataset is validation dataset
                                DatasetMapper(self.cfg,True)
                                ), 
                              eval_period = 20,
                              )
    )
    return hooks 