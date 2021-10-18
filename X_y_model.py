import mne
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

from skorch.callbacks import LRScheduler, BatchScoring

from braindecode.datasets import (
    create_from_mne_epochs, WindowsDataset, BaseConcatDataset)
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode import EEGRegressor


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


def create_dataset_new(fnames, ages):
    """

    """
    # TODO: switch to dataset creation like here?
    datasets = []
    for fname, age in zip(fnames, ages):
        # read epochs file
        epochs = mne.read_epochs(fname=fname)
        # fake metadata for braindecode. use window ids as well as age as target
        metadata = np.array([
            list(range(len(epochs))),  # i_window_in_trial (chunk of rec)
            len(epochs) * [-1],  # i_start_in_trial (unknown / unused)
            len(epochs) * [-1],  # i_stop_in_trial (unknown / unused
            len(epochs) * [age],  # target (subject age)
        ])
        metadata = pd.DataFrame(
            data=metadata.T,
            columns=['i_window_in_trial', 'i_start_in_trial', 'i_stop_in_trial',
                     'target'],
        )

        epochs.metadata = metadata
        # create a windows dataset
        ds = WindowsDataset(
            windows=epochs,
            description={'age': age, 'fname': fname},
            targets_from='metadata',
        )
        datasets.append(ds)
    return datasets


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
        The number of EEG channels in the mne.Epochs.
    """
    # insert the age of the subjects into the epochs events as description
    # this is where braindecode expects them
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
    windows_ds.set_description(description)
    return windows_ds, window_sizes[0], n_channels[0]


def create_model(model_name, window_size, n_channels, seed):
    """Create a braindecode model (either ShallowFBCSPNet or Deep4Net).

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
        model, n_epochs, batch_size, lr, weight_decay,
):
    """Create am estimator (EEGRegressor) that implements fit/transform.

    Parameters
    ----------
    model: braindecode.models.Deep4Net or braindecode.models.ShallowFBCSPNet
        A braindecode convolutional neural network.
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
    estimator: braindecode.EEGRegressor
        An estimator holding a braindecode model and implementing fit /
        transform.
    """
    estimator = EEGRegressor(
        model,
        criterion=torch.nn.L1Loss,  # optimize MAE
        optimizer=torch.optim.AdamW,
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
    return estimator


def get_estimator_and_X(
        fnames,
        ages,
        model_name,
        n_epochs,
        batch_size,
        seed,
):
    """Create am estimator (EEGRegressor) that implements fit/transform and a
    braindecode dataset.

    Parameters
    ----------
    fnames: list
        A list of .fif files to be used.
    ages: numpy.ndarray
        The subject ages corresponding to the recordings in the .fif files.
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    n_epochs: int
        The number of training epochs used in model training (required to
        in the creation of a learning rate scheduler).
    batch_size: int
        The size of training batches.
    seed: int
        The seed to be used to initialize the network.

    Returns
    -------
    model: braindecode.EEGRegressor
        A braindecode estimator implementing fit/transform.
    ds: braindecode.datasets.BaseConcatDataset
        A braindecode dataset holding all data and targets.
    """
    epochs, epoch_to_fif = read_epochs(fnames)
    ds, window_size, n_channels = create_dataset(
        epochs=[mne.read_epochs(fname) for fname in fnames],
        ages=ages,
        # for every pre-cut epoch, add an entry to description corresponding
        # to the recordings it originates from
        description={'rec': [j for i in epoch_to_fif for j in i]},
    )
    model, lr, weight_decay = create_model(
        model_name=model_name,
        window_size=window_size,
        n_channels=n_channels,
        seed=seed,
    )
    model = create_estimator(
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )
    return model, ds


def get_scores(y_true, y_pred, metrics):
    return {metric.__name__: metric(y_true, y_pred) for metric in metrics}


def cross_val_score(
    estimator,
    X,
    cv,
    fit_params,
):
    """Create am estimator (EEGRegressor) that implements fit/transform.

    Parameters
    ----------
    estimator: braindecode.EEGRegressor
        A braindecode estimator implementing fit/transform.
    X: braindecode.datasets.BaseConcatDataset
        A braindecode dataset holding all data and targets.
    cv: sklearn.model_selection.KFold
        A scikit-learn object to generate splits.
    fit_params: dict
        Additional parameters to be used when fitting the estimator.

    Returns
    -------
    scores: list
        A list holding mean absolute error and r2 score for all folds of cv.
    """
    scale_ages = True
    metrics = [mean_absolute_error, r2_score]
    scores = []
    # we split based on recordings, such that windows from one example
    # do NOT end up in both train and valid set
    n_recs = len(X.description['rec'].unique())
    for fold_i, (train_rec_i, valid_rec_i) in enumerate(cv.split(range(n_recs))):
        # we assign the windows into train and valid set according to the
        # recordings
        rec_splits = X.split('rec')
        train_set = BaseConcatDataset([rec_splits[str(i)] for i in train_rec_i])
        valid_set = BaseConcatDataset([rec_splits[str(i)] for i in valid_rec_i])
        # compute mean and std of train ages and set a target_transform to
        # both train and valid set
        train_ages = train_set.get_metadata().groupby('rec').head(1)['target']
        mean_train_rec_age = train_ages.mean()
        std_train_rec_age = train_ages.std()

        # scale the ages to zero mean unit variance
        if scale_ages:
            def scale_age(age):
                age = (age - mean_train_rec_age) / std_train_rec_age
                return age

            train_set.target_transform = scale_age
            valid_set.target_transform = scale_age

        # train the net
        estimator.fit(X=train_set, y=None, **fit_params)
        # we predict the validation set
        y_pred = estimator.predict(valid_set)
        df = valid_set.get_metadata()
        df['y_pred'] = y_pred

        # invert the target_transform
        if scale_ages:
            df['y_pred'] = df['y_pred'] * std_train_rec_age + mean_train_rec_age

        # these are window predictions and scores
        print(fold_i, 'window', get_scores(df['target'], df['y_pred'], metrics))

        # backtrack window predictions and generate recording predictions and
        # scores
        avg_df = df.groupby('rec').mean()
        this_scores = get_scores(avg_df['target'], avg_df['y_pred'], metrics)
        scores.append(this_scores)
        print(fold_i, 'rec', scores[-1])
    return scores
