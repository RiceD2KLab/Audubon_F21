import torch
from config import CONFIG, SEED, HYPERPARAMS, DEVICE, BIRD_ONLY
from config import DETECTOR_PATH, TILED_NEW_CSV_PATH, TILED_IMG_PATH
from src.data.utils import get_file_names, split_img_annos
from src.data.dataloader import get_od_dataloader
from src.data.transforms import get_transform
from src.models.pretrained import get_pretrained_od_model
from src.optimizers.sgd import get_sgd_optim
from train import train_detector


# Random seed
torch.manual_seed(SEED)


def train_pipeline(csv_path, img_path, split_ratio, batch_size, num_classes, l_r, num_epoch, model_name):
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
    print(results[0][-1])


train_pipeline(TILED_NEW_CSV_PATH, TILED_IMG_PATH,
               CONFIG['data_split'], CONFIG['batch_size'], CONFIG['model'][1],
               HYPERPARAMS['l_r'], HYPERPARAMS['num_epoch'], CONFIG['model'][0])
