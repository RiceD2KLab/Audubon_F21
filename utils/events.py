import wandb
from detectron2.utils.events import EventWriter, get_event_storage

class WAndBWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    Code adapted from: https://github.com/facebookresearch/detectron2/issues/774#issuecomment-776944522
    """

    def __init__(self, window_size: int = 20):
        self._window_size = window_size

    def write(self):
        storage = get_event_storage()
        stats = {}
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            stats[k.replace("/", "-")] = v[0]
        wandb.log(stats, step=storage.iter)

        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:\
                wandb.log({img_name: img}, step=step_num)

    def close(self):
        pass