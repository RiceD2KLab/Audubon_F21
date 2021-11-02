import numpy as np
import itertools
import detectron2.utils.comm as comm
from pycocotools.cocoeval import COCOeval
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.evaluation.coco_evaluation import COCOEvaluator

class PrecisionRecallEvaluator(COCOEvaluator):
    """
    Evaluate all Precision-Recall metrics for instance detection obtained from detectron2 COCOEvalulator
    """
    def __init__(self,dataset_name,output_dir=None):
        super().__init__(dataset_name, output_dir=output_dir)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        Outputs: (see COCOEval API for details)
            precisions  - [TxRxKxAxM] precision for every evaluation setting
            recalls    - [TxKxAxM] max recall for every evaluation setting

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

        coco_eval = self._coco_eval_predictions(predictions, img_ids=img_ids)

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return ([],[])

        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]  # precision has dims (iou, recall, cls, area range, max dets)
        assert len(self._metadata.get("thing_classes")) == precisions.shape[2]
        recalls = coco_eval.eval["recall"]
        assert len(self._metadata.get("thing_classes")) == recalls.shape[1]

        return (precisions, recalls)

    def _coco_eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions.
        """
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        coco_dt = self._coco_api.loadRes(coco_results)
        coco_eval = (COCOeval_opt if self._use_fast_impl else COCOeval)(self._coco_api, coco_dt, "bbox")
        if img_ids is not None:
            coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()

        return coco_eval









