import mne
import torch
import numpy as np
from sklearn.model_selection import KFold
from skorch.callbacks import LRScheduler, BatchScoring
from skorch.helper import predefined_split

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.datasets import create_from_mne_epochs
from braindecode import EEGRegressor


# TODO: add n_jobs somewhere?


def read_epochs(fnames):
    """Read all epochs .fif files from given fnames. For every .fif file, count
    how many epochs belong to it and return this as a nested list. We need this
    information in the end to go from epoch to fif predictions.

    Parameters
    ----------
    fnames: list
        A list of .fif files.

    Returns
    -------
    epochs: list
        A list of mne.Epochs objects with preload=False.
    epoch_to_fif: list
        A nested list of same length as fnames, where each entry is a list that
        has as many entries as there are epochs in the .fif file at the same
        position in fnames. Required after prediction to go from epoch to .fif
        predictions.
    """
    epochs = [mne.read_epochs(fname, preload=False) for fname in fnames]
    epoch_to_fif = [len(e) * [i] for i, e in enumerate(epochs)]
    return epochs, epoch_to_fif


def get_split_ids(n_examples, cv, fold):
    """Split n_examples to fold given strategy cv.

    Parameters
    ----------
    n_examples: int
        The number of examples in the dataset.
    cv: sklearn.model_selection.KFold
        A scikit-learn object to generate splits (e.g. KFold).
    fold: int
        The id of the fold to be used in model training.

    Returns
    -------
    train_is: list
        Integer ids of examples that belong to the train set.
    valid_is: list
        Integer ids of examples that belong to the valid set.
    """
    # split n_examples into train and valid with given cv strategy until
    # desired fold
    for fold_i, (train_is, valid_is) in enumerate(
            cv.split(list(range(n_examples)))):
        if fold_i == fold:
            break
    return train_is, valid_is


def split(to_split, train_is, valid_is):
    """Split something into train and valid.

    Parameters
    ----------
    to_split: array-like
        A list of things to be split with respect to train_is and valid_is.

    Returns
    -------
    train_split: list
        Examples of to_split at position train_is.
    valid_split: list
        Examples of to_split at position valid_is.
    """
    train_split = [to_split[i] for i in train_is]
    valid_split = [to_split[i] for i in valid_is]
    return train_split, valid_split


def create_datasets(epochs, ages, epoch_to_fif, train_is, valid_is, scale_ages):
    """Load epochs data stored as .fif files. Expects all .fif files to have
    epochs of equal length and to have equal channels.

    Parameters
    ----------
    epochs: list
        A list of mne.Epochs objects with preload=False.
    ages: list
        The subject ages corresponding to the recordings in the .fif files.
    epoch_to_fif: list
        A nested list of same length as fnames, where each entry is a list that
        has as many entries as there are epochs in the .fif file at the same
        position in fnames. Required after prediction to go from epoch to .fif
        predictions.
    train_is: list
        Integer ids of examples that belong to the train set.
    valid_is: list
        Integer ids of examples that belong to the valid set.
    scale_ages: bool
        Whether to scale ages to zero mean unit variance.

    Returns
    -------
    train_set: braindecode.datasets.BaseConcatDataset
        The training set.
    valid_set: braindecode.datasets.BaseConcatDataset
        The validation set.
    window_size: int
        The size of a single individual epoch.
    n_channels: int
        The number of channels in the epochs.
    """
    epochs_train, epochs_valid = split(epochs, train_is, valid_is)
    train_ages, valid_ages = split(ages, train_is, valid_is)
    epoch_to_fif_train, epoch_to_fif_valid = split(
        epoch_to_fif, train_is, valid_is)
    # flatten the lists
    epoch_to_fif_train = [j for i in epoch_to_fif_train for j in i]
    epoch_to_fif_valid = [j for i in epoch_to_fif_valid for j in i]
    train_set, window_size, n_channels = create_dataset(
        epochs=epochs_train,
        ages=train_ages,
        description={'fif': epoch_to_fif_train},
    )
    valid_set, window_size, n_channels = create_dataset(
        epochs=epochs_valid,
        ages=valid_ages,
        description={'fif': epoch_to_fif_valid},
    )
    # optionally, scale the ages to zero mean, unit variance
    # therefore, compute the mean and std on train ages
    # and set a target transform to both train and validation set
    if scale_ages:
        mean_train_age = np.mean(train_ages)
        std_train_age = np.std(train_ages)

        # define a target transform that first subtracts mean train age from
        # given age and then devides by std of train ages
        def scale_age(age):
            age = (age - mean_train_age) / std_train_age
            return age

        train_set.target_transform = scale_age
        valid_set.target_transform = scale_age
    return train_set, valid_set, n_channels, window_size


def create_dataset(epochs, ages, description):
    """Load epochs data stored as .fif files. Expects all .fif files to have
    epochs of equal length and to have equal channels.

    Parameters
    ----------
    epochs: list
        A list of mne.Epochs objects with preload=False.
    ages: numpy.ndarray
        The subject ages corresponding to the recordings in the .fif files.
    description: dict
        Information to add to dataset.description.

    Returns
    -------
    windows_ds: braindecode.datasets.BaseConcatDataset
        A dataset compatible with training of braindecode models via skorch.
    window_size: int
        The size of a single individual epoch.
    n_channels: int
        The number of channels in the epochs.
    """
    # insert the age of the subjects into the epochs events as description
    # this is where braindecode expects them
    # TODO: add to epochs.metadata instead?
    for i in range(len(epochs)):
        epochs[i].events[:, -1] = len(epochs[i]) * [ages[i]]
    # make sure we do not have a mess of window lengths / number of chs
    # therefore, load a single window of every epochs file and check its shape
    window_sizes = [e.get_data(item=0).shape[-1] for e in epochs]
    assert len(set(window_sizes)) == 1
    n_channels = [e.get_data(item=0).shape[-2] for e in epochs]
    assert len(set(n_channels)) == 1
    # create a braindecode WindowsDataset that features lazy loading and is
    # compatible with training of braindecode models via skorch.
    # assuming we obtain pre-cut trials, with the following line we are limited
    # to do trialwise decoding, since we set window_size to the length of the
    # trial. it could be beneficial to use an actual window_size smaller then
    # the trial length and to run cropped decoding (requires adjustment of the
    # loss computation etc).
    # if mne logging is enabled, it dramatically slows down the call below
    windows_ds = create_from_mne_epochs(
        list_of_epochs=epochs,
        window_size_samples=window_sizes[0],
        window_stride_samples=window_sizes[0],
        drop_last_window=False,
    )
    # TODO: set targets here?
    # windows_ds.set_description({'target': ages})
    windows_ds.set_description(description)
    return windows_ds, window_sizes[0], n_channels[0]


def create_model(model_name, window_size, n_channels, seed):
    """Create a braindecode model (either ShallowFBCSPNet or Deep4Net)

    Parameters
    ----------
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    window_size: int
        The length of the input data time series in samples.
    n_channels: int
        The number of input data channels.
    seed: int
        The seed to be used to initialize the network.

    Returns
    -------
    model: braindecode.models.Deep4Net or braindecode.models.ShallowFBCSPNet
        A braindecode convolutional neural network.
    lr: float
        The learning rate to be used in network training.
    weight_decay: float
        The weight decay to be used in network training.
    """
    # check if GPU is available, if True chooses to use it
    cuda = torch.cuda.is_available()
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


def create_estimator(
        model, train_set, valid_set, n_epochs, batch_size, lr, weight_decay,
):
    """Create am estimator (EEGRegressor) that implements fit/transform.

    Parameters
    ----------
    model: braindecode.models.Deep4Net or braindecode.models.ShallowFBCSPNet
        A braindecode convolutional neural network.
    train_set: braindecode.datasets.BaseConcatDataset
        The training set.
    valid_set: braindecode.datasets.BaseConcatDataset
        The validation set.
    n_epochs: int
        The number of training epochs used in model training (required to
        in the creation of a learning rate scheduler).
    batch_size: int
        The size of training batches.
    lr: float
        The learning rate to be used in network training.
    weight_decay: float
        The weight decay to be used in network training.

    Returns
    -------
    X: braindecode.datasets.BaseConcatDataset
        The training set to be used in model.fit(X=X, y=y)
    y: None
        Exists for compatibility only. Actual target is included in the train
        and valid set.
    model: braindecode.EEGRegressor
        An estimator holding a braindecode model and implementing fit /
        transform.
    """
    clf = EEGRegressor(
        model,
        criterion=torch.nn.L1Loss,  # optimize MAE
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        # using valid_set for validation
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            # ("R2", BatchScoring('r2', lower_is_better=False)),
            # ("MAE", BatchScoring("neg_mean_absolute_error",
            #                      lower_is_better=False)),
            ("lr_scheduler", LRScheduler('CosineAnnealingLR',
                                         T_max=n_epochs-1)),
        ],
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    # y is None, since the train_set returns x, y, ind when iterrated, all that
    # is needed for training to work
    # training can be performed by 'clf.fit(X=train_set, y=y, epochs=n_epochs)'
    return train_set, valid_set, clf


def get_train_valid_model(
        fnames, ages, model_name, cv, fold, n_epochs=35, batch_size=64,
        seed=20211011,
):
    """Create am estimator (EEGRegressor) that implements fit/transform.

    Parameters
    ----------
    fnames: list
        A list of .fif files to be used.
    ages: numpy.ndarray
        The subject ages corresponding to the recordings in the .fif files.
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    cv: sklearn.model_selection.KFold
        A scikit-learn object to generate splits (e.g. KFold).
    fold: int
        The id of the fold to be used in model training.
    n_epochs: int
        The number of training epochs used in model training (required to
        in the creation of a learning rate scheduler).
    batch_size: int
        The size of training batches.
    seed: int
        The seed to be used to initialize the network.

    Returns
    -------
    X: braindecode.datasets.BaseConcatDataset
    y: None
        Exists for compatibility only. Actual target is included in the train
        and valid set.
    model: braindecode.EEGRegressor
        An estimator holding a braindecode model and implementing fit /
        transform.
    """
    epochs, epoch_to_fif = read_epochs(fnames)
    train_is, valid_is = get_split_ids(
        n_examples=len(fnames),
        cv=cv,
        fold=fold,
    )
    train_set, valid_set, n_channels, window_size = create_datasets(
        epochs=epochs,
        ages=ages,
        epoch_to_fif=epoch_to_fif,
        train_is=train_is,
        valid_is=valid_is,
        scale_ages=True,
    )
    model, lr, weight_decay = create_model(
        model_name=model_name,
        window_size=window_size,
        n_channels=n_channels,
        seed=seed,
    )
    return create_estimator(
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )
