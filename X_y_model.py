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


# if mne logging is enabled it dramatically slows down this function
def create_windows_ds(fnames, ages):
    """Load epochs data stored as .fif files. Expects all .fif files to have
    epochs of equal length and to have equal channels.

    Parameters
    ----------
    fnames: list
        A list of .fif files to be used.
    ages: numpy.ndarray
        The subject ages corresponding to the recordings in the .fif files.

    Returns
    -------
    windows_ds: braindecode.datasets.BaseConcatDataset
        A dataset compatible with training of braindecode models via skorch.
    window_size: int
        The size of a single individual epoch.
    n_channels: int
        The number of channels in the epochs.
    window_map: list
        TODO
    """
    # read all the epochs fif files
    epochs = [mne.read_epochs(fname, preload=False) for fname in fnames]
    window_map = [i for i, e in enumerate(epochs) for _ in range(len(e.events))]
    print('window_map', window_map, len(window_map))
    assert len(epochs) == len(ages)
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
    windows_ds = create_from_mne_epochs(
        list_of_epochs=epochs,
        window_size_samples=window_sizes[0],
        window_stride_samples=window_sizes[0],
        drop_last_window=False,
    )
    # TODO: set targets here?
    # windows_ds.set_description({'target': ages})
    return windows_ds, window_sizes[0], n_channels[0], window_map


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


def create_data_split(windows_ds, cv, fold, window_map):
    """Split the dataset into train and validation following cv and return data
    with respect to specified fold.

    Parameters
    ----------
    windows_ds: braindecode.datasets.BaseConcatDataset
        The dataset to be split into train and valid.
    cv: sklearn.model_selection.KFold
        A scikit-learn object to generate splits (e.g. KFold).
    fold: int
        The id of the fold to be used in model training.
    window_map:

    Returns
    -------
    train_set: braindecode.datasets.BaseConcatDataset
        The training set.
    valid_set: braindecode.datasets.BaseConcatDataset
        The validation set.
    """
    # TODO: there might be a better way to perform cv. check out skorch
    # we already need split data to initialize the EEGRegressor, since we want
    # to give it a predefined validation set.
    # therefore, use the input arguments cv and fold to determine the split
    # ids here and split the data accordingly
    example_ids = np.arange(len(windows_ds))
    print('example_ids', example_ids)
    for fold_i, (train_is, valid_is) in enumerate(cv.split(example_ids)):
        if fold_i == fold:
            break
    # the following is only valid because we are doing trialwise decoding
    # which means that every compute window represents exactly one trial. if
    # this was not the case, so if we used a window_size different from trial
    # length in create_windows_ds, splitting here could cause that windows of
    # a single trial end up in both train and valid set.
    train_set = windows_ds.split(by=train_is)['0']
    valid_set = windows_ds.split(by=valid_is)['0']
    # add target transform to scale ages to zero mean, unit variance

    """
    TODO:
    print("metadata ages", windows_ds.get_metadata()['target'].values)
    print("ages", windows_ds.get_metadata()[
        'target', np.array(window_map)[train_is]].values)

    def scale_age(age, mean_train_age=np.mean(train_ages),
                  std_train_age=np.std(train_ages)):
        age = (age - mean_train_age) / std_train_age
        return age

    train_set.target_transform = scale_age
    valid_set.target_transform = scale_age
    """
    return train_set, valid_set


def create_estimator(
        model_name, train_set, valid_set, n_channels, window_size, n_epochs,
        batch_size, seed,
):
    """Create am estimator (EEGRegressor) that implements fit/transform.

    Parameters
    ----------
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    train_set: braindecode.datasets.BaseConcatDataset
        The training set.
    valid_set: braindecode.datasets.BaseConcatDataset
        The validation set.
    n_channels: int
        The number of input data channels.
    window_size: int
        The length of the input data time series in samples.
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
        The training set to be used in model.fit(X=X, y=y)
    y: None
        Exists for compatibility only. Actual target is included in the train
        and valid set.
    model: braindecode.EEGRegressor
        An estimator holding a braindecode model and implementing fit /
        transform.
    """
    model, lr, weight_decay = create_model(
        model_name=model_name,
        window_size=window_size,
        n_channels=n_channels,
        seed=seed,
    )
    # using BatchScoring over strings did not enable usage of sklearn functions
    # like cross_val_score with the EEGRegressor
    from skorch.callbacks import BatchScoring
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
            ("R2", BatchScoring('r2', lower_is_better=False)),
            #  ("MAE", BatchScoring("neg_mean_absolute_error", lower_is_better=False)),
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs-1)),
        ],
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    # y is None, since the train_set returns x, y, ind when iterrated, all that
    # is needed for training to work
    # training can be performed by 'clf.fit(X=train_set, y=y, epochs=n_epochs)'
    # TODO: partially inititalize 'fit' with 'epochs=n_epochs', such that the
    # call is identical to other estimators?
    return train_set, None, clf


def get_X_y_model(
        fnames, model_name, ages, cv, fold, n_epochs=35, batch_size=64,
        seed=20211011,
):
    """Create am estimator (EEGRegressor) that implements fit/transform.

    Parameters
    ----------
    fnames: list
        A list of .fif files to be used.
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    ages: numpy.ndarray
        The subject ages corresponding to the recordings in the .fif files.
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
    windows_ds, n_channels, window_size, window_map = create_windows_ds(
        fnames=fnames,
        ages=ages,
    )
    train_set, valid_set = create_data_split(
        windows_ds=windows_ds,
        cv=cv,
        fold=fold,
        window_map=window_map,
    )
    return create_model_and_data_split(
        model_name=model_name,
        train_set=train_set,
        valid_set=valid_set,
        n_channels=n_channels,
        window_size=window_size,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
    )
