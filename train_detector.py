import torch
from config import CONFIG_DETECTOR, SEED, HYPERPARAMS_DETECTOR, DEVICE, BIRD_ONLY
from config import DETECTOR_PATH, TILED_NEW_CSV_PATH, TILED_IMG_PATH, PLOTS_PATH, DPI
from src.data.utils import get_file_names, split_img_annos
from src.data.dataloader import get_od_dataloader
from src.data.transforms import get_transform
from src.data.plotlib import plot_curves, plot_precision_recall, visualize_predictions
from src.models.pretrained import get_pretrained_od_model
from src.optimizers.sgd import get_sgd_optim
from src.train import train_detector
from src.eval import get_od_predictions


# Random seed
torch.manual_seed(SEED)


def train_detector_pipeline(csv_path, img_path, split_ratio, batch_size, num_classes, l_r, num_epoch, model_name):
    ''' Train a detector model using the given hyperparameters and configurations. '''
    # Add JPG and CSV file names to a dictionary (makes referencing the files easier).
    csv_files = get_file_names(csv_path, 'csv')
    jpg_files = get_file_names(img_path, 'jpg')

    # Split the dataset into training set, test set, and validation set.
    trainset, testset, valset = split_img_annos(jpg_files, csv_files, split_ratio, seed=SEED)

    # Dataloaders
    trainloader = get_od_dataloader(
        trainset['jpg'], trainset['csv'],
        get_transform(train=True), batch_size,
        True, BIRD_ONLY
    )

    valloader = get_od_dataloader(
        valset['jpg'], valset['csv'],
        get_transform(train=False), batch_size,
        False, BIRD_ONLY
    )

    model = get_pretrained_od_model(num_classes)
    optimizer = get_sgd_optim(model, l_r)

    results = train_detector(
        model,
        optimizer,
        None,  # TODO: add customized loss function
        num_epoch,
        trainloader,
        valloader,
        DEVICE,
        DETECTOR_PATH,
        model_name,
        print_every=5
    )

    plot_curves(results[0], results[1], 'training loss', 'validation loss', 'epoch', 'loss',
                f'Training and validation loss curves of {model_name} detector', PLOTS_PATH)
    plot_precision_recall(results[2], 'epoch', 'precision and recall',
                          f'Training precision and recall curves of {model_name} detector', PLOTS_PATH)
    plot_precision_recall(results[3], 'epoch', 'precision and recall',
                          f'Validation precision and recall curves of {model_name} detector', PLOTS_PATH)

    batch = 0
    preds = get_od_predictions(model, valloader, DEVICE, batch)
    for idx in range(len(preds)):
        visualize_predictions(valset['jpg'][idx + batch * batch_size],
                              preds[idx], PLOTS_PATH, model_name + '_batch_' + str(batch) + '_idx_' + str(idx),
                              DPI, 0.5)


if __name__ == '__main__':
    train_detector_pipeline(TILED_NEW_CSV_PATH, TILED_IMG_PATH,
                            CONFIG_DETECTOR['data_split'], CONFIG_DETECTOR['batch_size'], CONFIG_DETECTOR['model'][1],
                            HYPERPARAMS_DETECTOR['l_r'], HYPERPARAMS_DETECTOR['num_epoch'], CONFIG_DETECTOR['model'][0])
