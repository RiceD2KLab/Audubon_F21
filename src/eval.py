import torch
from tqdm import tqdm
from .data.coco.coco_utils import get_coco_api_from_dataset
from .data.coco.coco_eval import CocoEvaluator


def get_od_predictions(model, dataloader, device, idx):
    ''' Returns object detection results for a given index in a dataloader '''
    model.eval()
    with torch.no_grad():
        for batch_id, (images, targets) in enumerate(dataloader):
            if batch_id == idx:
                # move data to device
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # forward pass
                outputs = model(images)
                return outputs
    return None


def get_od_loss(model, loss_fn, dataloader, device):
    ''' Returns loss for an object detection model on a given dataset '''
    loss = 0
    with torch.no_grad():
        for batch_id, (images, targets) in enumerate(tqdm(dataloader, leave=True, ncols=80)):
            # move data to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss += losses.item()
    loss = loss / len(dataloader)
    return loss


def get_od_stats(model, dataloader, device):
    ''' Returns stats for an object detection model on a given dataset '''
    n_threads = torch.get_num_threads()
    print(f"Using {n_threads} threads for inference")
    torch.set_num_threads(n_threads)

    cpu_device = torch.device("cpu")
    model.eval()
    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            images = list(img.to(device) for img in images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outputs = model(images)
            outputs = [{key: val.to(cpu_device) for key, val in out.items()} for out in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats

    return stats
