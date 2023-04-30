import torch
from config import CLASSIFIER_PATH, CONFIG_CLASSIFIER, HYPERPARAMS_CLASSIFIER
from config import CLF_TRAIN_PATH, CLF_VAL_PATH, CROPPED_PATH, DEVICE, PLOTS_PATH, DATA_PATH
import torchvision.datasets as datasets
from src.data.dataloader import get_clf_dataloader_from_dir
from torchvision.models import ResNet50_Weights
from src.models.pretrained import get_pretrained_resnet50
from src.optimizers.adam import get_adam_optim
from src.loss_fn.weighted_cross_entropy import compute_class_weights_from_dataset, get_weighted_cross_entropy_loss_fn
from src.train import train_classifier
from src.eval import get_clf_predictions, get_stats_from_confusion_matrix
from src.data.plotlib import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from src.data.plotlib import plot_curves


def train_classifier_pipline(all_data_dir, train_dir, val_dir, batch_size, n_epochs, name, save_path, device, lr):
    ''' Train a classifier model using the given hyperparameters and configurations.'''
    # explore data
    all_data = datasets.ImageFolder(all_data_dir)
    all_dataloader = get_clf_dataloader_from_dir(all_data_dir, batch_size=batch_size, shuffle=False, preprocess=None)
    class_names = all_dataloader.dataset.classes

    # weights and tranformations
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()

    # dataloaders
    trainloader = get_clf_dataloader_from_dir(train_dir, batch_size=batch_size, shuffle=True, preprocess=preprocess)
    valloader = get_clf_dataloader_from_dir(val_dir, batch_size=batch_size, shuffle=False, preprocess=preprocess)

    # get resnet50 model
    model = get_pretrained_resnet50(num_classes=len(class_names), weights=weights)

    # get optimizer and weighted cross entropy loss function
    optimizer = get_adam_optim(model, lr=lr)
    class_weights = compute_class_weights_from_dataset(all_data)
    loss_fn = get_weighted_cross_entropy_loss_fn(class_weights, device=device)

    # train classifier
    results = train_classifier(model, optimizer, loss_fn, n_epochs,
                               trainloader, valloader, device, save_path, name, print_every=5)

    plot_curves(results[0], results[1], 'training loss', 'validation loss', 'epoch', 'loss',
                f'Training and validation loss curves of {name} bird classifier', PLOTS_PATH)

    # load the best classifier
    model = torch.load(save_path + name + '.pt')

    true_labels, predicted = get_clf_predictions(model, valloader, device)
    true_labels_list = torch.concat(true_labels).tolist()
    predicted_list = torch.concat(predicted).tolist()

    # plot confusion matrix
    _ = plot_confusion_matrix(true_labels_list, predicted_list, class_names,
                              title=f'Confusion matrix of {name} bird classifier on validation set',
                              path=PLOTS_PATH)

    # get stats
    conf_mat = confusion_matrix(true_labels_list, predicted_list)
    stats = get_stats_from_confusion_matrix(conf_mat, class_names)
    stats.to_csv(DATA_PATH + name + '_stats.csv', index=False)


if __name__ == '__main__':
    train_classifier_pipline(CROPPED_PATH, CLF_TRAIN_PATH, CLF_VAL_PATH,
                             CONFIG_CLASSIFIER['batch_size'], HYPERPARAMS_CLASSIFIER['num_epoch'],
                             CONFIG_CLASSIFIER['model'], CLASSIFIER_PATH, DEVICE, HYPERPARAMS_CLASSIFIER['l_r'])
