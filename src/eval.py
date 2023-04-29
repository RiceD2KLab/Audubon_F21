import torch
from .train.coco.coco_utils import get_coco_eval_api_from_dataset
from .train.coco.coco_eval import CocoEvaluator


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
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch_id, (images, targets) in enumerate(dataloader):
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
    '''Returns stats for an object detection model on a given dataset'''
    # parallelize evaluation
    n_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    # switch model to evaluate mode
    model.eval()

    # get object detection evaluation api
    od_eval_api = get_coco_eval_api_from_dataset(dataloader.dataset)

    # set evaluation types
    iou_types = ["bbox"]

    # get object detection evaluator
    od_evaluator = CocoEvaluator(od_eval_api, iou_types)

    # evaluate
    with torch.no_grad():
        for batch_id, (images, targets) in enumerate(dataloader):
            # TODO: evaluate on batches of images
            pass

    stats = od_evaluator.get_stats()
    return stats
