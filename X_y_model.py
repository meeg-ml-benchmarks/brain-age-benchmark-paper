import mne
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

from skorch.callbacks import LRScheduler, BatchScoring
from skorch.helper import SliceDataset

from braindecode.datasets import (
    create_from_mne_epochs, WindowsDataset, BaseConcatDataset)
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.training.scoring import parse_callbacks
from braindecode import EEGRegressor


class CustomSliceDataset(SliceDataset):
    """A modified skorch.helper.SliceDataset to cast singe integers to valid
    2-dimensional scikit-learn regression targets.
    """
    # y has to be 2 dimensional, so call y.reshape(-1, 1)
    def __init__(self, dataset, idx=0, indices=None):
        super().__init__(dataset=dataset, idx=idx, indices=indices)

    def __getitem__(self, i):
        item = super().__getitem__(i)
        return np.array(item).reshape(-1, 1)


class BraindecodeKFold(KFold):
    """An adapted sklearn.model_selection.KFold that gets braindecode datasets
    of length n_compute_windows but splits based on the number of original
    files.
    """
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def split(self, X, y=None, groups=None):
        # split recordings instead of windows
        split = super().split(X=X.dataset.datasets)
        rec = X.dataset.get_metadata()['rec']
        # the index of DataFrame rec now corresponds to the id of windows
        rec = rec.reset_index()
        for train_i, valid_i in split:
            # print(len(train_i), len(valid_i))
            # map recording ids to window ids
            train_window_i = [j for i in train_i for j in
                              rec[rec['rec'] == i].index.to_list()]
            valid_window_i = [j for i in valid_i for j in
                              rec[rec['rec'] == i].index.to_list()]
            # print(len(train_window_i), len(valid_window_i))
            yield train_window_i, valid_window_i


class RecScorer():
    """Compute recording scores by averaging all window predictions and labels
     of a recording."""
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, estimator, X, y):
        y_pred = estimator.predict(X)
        df = X.dataset.get_metadata()
        df.reset_index(inplace=True, drop=True)
        # get metadata of valid_set
        df = df.iloc[X.indices]
        df['y_true'] = y
        df['y_pred'] = y_pred
        # create rec scores
        df = df.groupby('rec').mean()
        return self.metric(y_true=df['y_true'], y_pred=df['y_pred'])


def create_dataset(fnames, ages):
    """Read all epochs .fif files from given fnames. Convert to braindecode
    dataset and add ages as targets.

    Parameters
    ----------
    fnames: list
        A list of .fif files.
    ages: array-like
        Subject ages.

    Returns
    -------
    braindecode.datasets.BaseConcatDataset
        A braindecode dataset.
    """
    datasets = []
    for rec_i, (fname, age) in enumerate(zip(fnames, ages)):
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
        # no idea why this is necessary but without the metadata dataframe had
        # an index like 4,7,8,9, ... which caused an KeyError on getitem through
        # metadata.loc[idx]. resetting index here fixes that
        epochs.metadata.reset_index(drop=True, inplace=True)
        # create a windows dataset
        ds = WindowsDataset(
            windows=epochs,
            description={'age': age, 'fname': fname, 'rec': rec_i},
            targets_from='metadata',
        )
        datasets.append(ds)
    ds = BaseConcatDataset(datasets)
    x, y, ind = ds[0]
    return ds, x.shape[-2], x.shape[-1]


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
    callbacks = parse_callbacks(
        callbacks=[
            # ("R2", BatchScoring('r2', lower_is_better=False)),
            # ("MAE", BatchScoring("neg_mean_absolute_error",
            #                      lower_is_better=False)),
            ("lr_scheduler", LRScheduler('CosineAnnealingLR',
                                         T_max=n_epochs-1)),
        ],
        cropped=False,
    )
    estimator = EEGRegressor(
        model,
        criterion=torch.nn.L1Loss,  # optimize MAE
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=callbacks,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    return estimator


def X_y_model(
        fnames,
        ages,
        model_name,
        n_epochs,
        batch_size,
        seed,
):
    """Create am estimator (EEGRegressor) that implements fit/transform and a
    braindecode dataset that returns X and y.

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
    X: skorch.helper.SliceDataset
        A dataset that gives X.
    y: skorch.helper.SliceDataset
        A modified SliceDataset that gives ages reshaped to (-1, 1).
    estimator: braindecode.EEGRegressor
        A braindecode estimator implementing fit/transform.
    """
    ds, n_channels, window_size = create_dataset(
        fnames=fnames,
        ages=ages,
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
    X = SliceDataset(ds, idx=0)
    y = CustomSliceDataset(ds, idx=1)
    return X, y, model
