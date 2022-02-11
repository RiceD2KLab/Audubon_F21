import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import cv2
import itertools
import detectron2.utils.comm as comm
from pycocotools.cocoeval import COCOeval
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader


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


def get_precisions_recalls(cfg, predictor, dataset_name):
    """
    get precisions and recalls outputted by PrecisionRecallEvaluator
    INPUTS:
        cfg -- detectron2 CfgNode
        predictor -- detectron2 predictor
        dataset_name -- registered dataset name to evaluate on
    """
    evaluator = PrecisionRecallEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    data_loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(predictor.model, data_loader, evaluator)


def plot_precision_recall(precisions, max_recalls, class_names, class_colors):
    """
    Plot precision-recall curves outputted by PrecisionRecallEvaluator
    INPUTS:
         precisions: [TxRxKxAxM] precision for every evaluation setting
         max_recalls: [TxKxAxM] max recall for every evaluation setting
         class_names: names of bird species registered.
         class_colors: List of colors for corresponding to bird species
    """
    recall = np.linspace(0, 1, 101)
    fig, ax = plt.subplots()
    fig_iou50, ax_iou50 = plt.subplots()
    fig_iou75, ax_iou75 = plt.subplots()
    for c_indx, class_name in enumerate(class_names):
        avg_precision = np.mean(np.squeeze(precisions[:, :, c_indx, 0, -1]), axis=1)
        print(f"AP50 for {class_name}: {avg_precision[0]}")
        print(f"AP75 for {class_name}: {avg_precision[5]}")
        max_recall = np.squeeze(max_recalls[:, c_indx, 0, -1])
        print(f"Max recall (IoU 50) for {class_name}: {max_recall[0]}")
        ax.plot(max_recall, avg_precision, color=class_colors[c_indx])
        precisions_iou50 = np.squeeze(precisions[0, :, c_indx, 0, -1])
        ax_iou50.plot(recall, precisions_iou50, color=class_colors[c_indx])
        precisions_iou75 = np.squeeze(test_precisions[5, :, c_indx, 0, -1])
        ax_iou75.plot(recall, precisions_iou75, color=class_colors[c_indx])

    ax.set(ylabel="Avg. Precision",
           xlabel="Max Recall")
    ax_iou50.set(title="Precision-Recall Curve (IoU = 0.5)",
                 ylabel="Precision",
                 xlabel="Recall")
    ax_iou75.set(title="Precision-Recall Curve (IoU = 0.75)",
                 ylabel="Precision",
                 xlabel="Recall")
    ax.legend(class_names, loc='best')
    ax_iou50.legend(class_names, loc='best')
    ax_iou75.legend(class_names, loc='best')

    plt.show()


def non_max_suppression_fast(df, overlap_thresh = 0.5):
    """
    Perform non-maximal supression for bounding boxes
    Adapted from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    INPUTS
        df -- pandas dataframe containing bounding boxes
        overlapThresh -- overlapping IoU threshold to be used for rejection (default: 0.5)
    OUTPUT
        df -- pandas dataframe containing bounding boxes after NMS
    """
    boxes = df['boxes']
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    pick = []
    x1 = np.array(boxes.map(lambda x: x[0]))
    y1 = np.array(boxes.map(lambda x: x[2]))
    x2 = np.array(boxes.map(lambda x: x[1]))
    y2 = np.array(boxes.map(lambda x: x[3]))
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(xx2 - xx1 + 1, 0)
        h = np.maximum(yy2 - yy1 + 1, 0)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return df.iloc[pick]


def evaluate_full_pipeline(eval_file_lst, predictor, species_map, raw_img_width, raw_img_height,
                           crop_width, crop_height, sliding_size):

    obj_dict = {'cnt_id': [], 'file_name': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'score': [],
                'pred_cls': []}
    idx = 1
    for f in tqdm(eval_file_lst):
        im = cv2.imread(f)
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = outputs["instances"].to("cpu")
        outputs = outputs[outputs.scores > 0.5]
        lst = list(outputs._fields.items())
        for ts in list(lst[0][1]):
            obj_dict['xmin'].append(ts.tolist()[0])
            obj_dict['ymin'].append(ts.tolist()[1])
            obj_dict['xmax'].append(ts.tolist()[2])
            obj_dict['ymax'].append(ts.tolist()[3])
            # add count_id to dict
            obj_dict['cnt_id'].append(idx)
            idx += 1
        # input score to dict
        score_for_file = lst[1][1].tolist()
        obj_dict['score'] = obj_dict['score'] + score_for_file
        # input predicted cls to dict
        class_for_file = lst[2][1].tolist()
        obj_dict['pred_cls'] = obj_dict['pred_cls'] + class_for_file
        obj_dict['file_name'] = obj_dict['file_name'] + [f] * len(score_for_file)

        output_df = pd.DataFrame(obj_dict)
        output_df['pred_cls'] = output_df['pred_cls'].map(species_map)

        # convert the tiled coordinates to original coordinates
        # ((img_width - crop_width) // sliding_size + 1) and i < ((img_height - crop_height) // sliding_size + 1)
        # get original coord idx from file name
        output_df['height_idx'] = output_df['file_name'].map(lambda x: int((os.path.split(x)[1].split('.')[0].split('_')[-2])))
        output_df['width_idx'] = output_df['file_name'].map(lambda x: int((os.path.split(x)[1].split('.')[0].split('_')[-1])))

        # get original file name
        output_df['orig_name'] = output_df['file_name'].map(
            lambda x: '_'.join(os.path.split(x)[1].split('_')[:-2]) + '.JPG')

        # convert xmin, xmax, ymin, ymax
        def convert_xmin(row):
            if row['width_idx'] == ((raw_img_width - crop_width) // sliding_size + 1):
                return raw_img_width - (crop_width - row['xmin'])
            else:
                return row['width_idx'] * sliding_size + row['xmin']

        def convert_xmax(row):
            if row['width_idx'] == ((raw_img_width - crop_width) // sliding_size + 1):
                return raw_img_width - (crop_width - row['xmax'])
            else:
                return row['width_idx'] * sliding_size + row['xmax']

        def convert_ymin(row):
            if row['height_idx'] == ((raw_img_height - crop_height) // sliding_size + 1):
                return raw_img_height - (crop_height - row['ymin'])
            else:
                return row['height_idx'] * sliding_size + row['ymin']

        def convert_ymax(row):
            if row['height_idx'] == ((raw_img_height - crop_height) // sliding_size + 1):
                return raw_img_height - (crop_height - row['ymax'])
            else:
                return row['height_idx'] * sliding_size + row['ymax']

        output_df['height_idx'] = output_df['file_name'].map(lambda x: int((os.path.split(x)[1].split('.')[0].split('_')[-2])))
        output_df['width_idx'] = output_df['file_name'].map(lambda x: int((os.path.split(x)[1].split('.')[0].split('_')[-1])))
        output_df['orig_xmin'] = output_df.apply(convert_xmin, axis=1, result_type='reduce')
        output_df['orig_xmax'] = output_df.apply(convert_xmax, axis=1, result_type='reduce')
        output_df['orig_ymin'] = output_df.apply(convert_ymin, axis=1, result_type='reduce')
        output_df['orig_ymax'] = output_df.apply(convert_ymax, axis=1, result_type='reduce')

        output_df['boxes'] = output_df[['orig_xmin', 'orig_xmax', 'orig_ymin', 'orig_ymax']].values.tolist()
        output_df = output_df.groupby('orig_name').apply(non_max_suppression_fast)

    return output_df



