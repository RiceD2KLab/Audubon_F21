from collections import OrderedDict
import logging, itertools
import detectron2.utils.comm as comm
from detectron2.evaluation.coco_evaluation import COCOEvaluator

# TODO: complete PrecisionRecallEvaluator


class PrecisionRecallEvaluator(COCOEvaluator):
    def __init__(self,dataset_name,output_dir=None):
        super(PrecisionRecallEvaluator, self).__init__(dataset_name, output_dir=output_dir)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _get_coco_precisions(self):

        return results






