import mne
import torch
import numpy as np
from sklearn.model_selection import KFold
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.datasets import create_from_mne_epochs
from braindecode import EEGRegressor


def create_windows_ds(fnames, ages):
    # read all the epochs fif files
    epochs = [mne.read_epochs(fname, preload=False) for fname in fnames]
    assert len(epochs) == len(ages)
    # insert the age of the subjects into the epochs events as decription
    # this is where braindecode expects them
    for i in range(len(epochs)):
        epochs[i].events[:, -1] = len(epochs[i]) * [ages[i]]
    # make sure we do not have a mess of window lengths / number of chs
    window_sizes = [e.get_data(item=0).shape[-1] for e in epochs]
    assert len(set(window_sizes)) == 1
    n_channels = [e.get_data(item=0).shape[-2] for e in epochs]
    assert len(set(n_channels)) == 1
    # create a braindecode WindowsDataset that features lazy loading and is
    # compatible with training of braindecode models via skorch.
    # assuming we obtain pre-cut trials, with the following line we are limited
    # to do trialwise decoding, since we set window_size to the length of the
    # trial. it could be beneficial to use an actual window_size smaller then
    # the trial length and to run cropped decoding.
    windows_ds = create_from_mne_epochs(
        list_of_epochs=epochs,
        window_size_samples=window_sizes[0],
        window_stride_samples=window_sizes[0],
        drop_last_window=False,
    )
    return windows_ds, window_sizes[0], n_channels[0]


def create_model(model_name, window_size, n_channels, seed):
    # check if GPU is available, if True chooses to use it
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    if model_name == 'shallow':
        model = ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=1,
            input_window_samples=window_size,
            final_conv_length='auto',
        )
        # TODO: insert age decoding hyperparams
        lr = 0.0625 * 0.01
        weight_decay = 0
    else:
        assert model_name == 'deep'
        model = Deep4Net(
            in_chans=n_channels,
            n_classes=1,
            input_window_samples=window_size,
            final_conv_length='auto',
        )
        # TODO: insert age decoding hyperparams
        lr = 1 * 0.01
        weight_decay = 0.5 * 0.001

    # remove the softmax layer from models
    new_model = torch.nn.Sequential()
    for name, module_ in model.named_children():
        if "softmax" in name:
            continue
        new_model.add_module(name, module_)
    model = new_model

    # Send model to GPU
    if cuda:
        model.cuda()
    return model, lr, weight_decay


def create_model_and_data_split(
        model_name, windows_ds, n_channels, window_size, n_epochs, cv, fold,
        seed, batch_size,
):
    model, lr, weight_decay = create_model(
        model_name=model_name,
        window_size=window_size,
        n_channels=n_channels,
        seed=seed,
    )

    # TODO: there might be a better way to perform cv. check out skorch
    # we already need split data to initialize the EEGRegressor
    # since we do 10 fold cv, use the fold input argument to determine the split
    # ids here again and split the data accordingly
    example_ids = np.arange(len(windows_ds))
    for fold_i, (train_is, valid_is) in enumerate(cv.split(example_ids)):
        if fold_i == fold:
            break
    train_set = windows_ds.split(by=train_is)['0']
    valid_set = windows_ds.split(by=valid_is)['0']

    clf = EEGRegressor(
        model,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        # using valid_set for validation
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "r2",
            "neg_mean_absolute_error",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs-1)),
        ],
        device=device,
    )
    # y is None, since the train_set returns x, y, ind when iterrated, all that
    # is needed for training to work
    # training can be performed by 'clf.fit(X=train_set, y=y, epochs=n_epochs)'
    return train_set, None, clf


def get_X_y_model(
        fnames, model_name, age, cv,  fold, n_epochs=35, batch_size=64,
        seed=20211011,
):
    windows_ds, n_channels, window_size = create_windows_ds(
        fnames=fnames,
        ages=age,
    )
    return create_model_and_data_split(
        model_name=model_name,
        windows_ds=windows_ds,
        n_channels=n_channels,
        window_size=window_size,
        cv=cv,
        fold=fold,
        seed=seed,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )







"""

def read_recordings(fnames):
    windows_ds_list, all_n_channels, all_window_sizes = [], [], []
    for fname in fnames:
        single_windows_ds, n_channels, window_size = read_recording(fname)
        windows_ds_list.append(single_windows_ds)
        all_n_channels.append(n_channels)
        all_window_sizes.append(window_size)
    # theoretically, every fif file in fnames could have a different
    # amount of time samples and a different amount of channels
    # make sure this is not the case
    assert len(set(all_n_channels)) == 1
    assert len(set(all_window_sizes)) == 1
    windows_ds = BaseConcatDataset(windows_ds_list)
    return windows_ds, all_n_channels[0], all_window_sizes[0]


def read_recording(fname):
    epochs = mne.read_epochs(fname, preload=False)
    # load a single window to know how many channels and time samples we are
    # working with
    n_channels, window_size = epochs.get_data(item=0).shape
    windows_ds = create_from_mne_epochs(
        list_of_epochs=[epochs],
        window_size_samples=window_size,
        window_stride_samples=window_size,
        drop_last_window=False,
    )
    return windows_ds, window_size, n_channels
"""
