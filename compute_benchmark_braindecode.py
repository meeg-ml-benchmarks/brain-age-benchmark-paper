# %% imports
import sys
from glob import glob

import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

from X_y_model import X_y_model, BraindecodeKFold as KFold, RecScorer

participants_path = sys.argv[1]  # '/home/lukas/Code/data/bids/participants.tsv'
data_path = sys.argv[2]   # '/home/lukas/Code/data/bids_deriv/'
subjects_df = pd.read_csv(participants_path, sep='\t')
fif_paths = glob(data_path+'/**/*clean_epo.fif', recursive=True)
subjects_df['fif_path'] = fif_paths

ages = subjects_df['age'].values
fnames = subjects_df['fif_path'].values

n_epochs = 35
batch_size = 64
seed = 20211012
n_jobs = 4
n_splits = 10
shuffle = True
random_state = 42
metrics = [mean_absolute_error, r2_score]

cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
scoring = {}
for m in metrics:
    scoring.update({'_'.join(['window', m.__name__]): make_scorer(m)})
    scoring.update({'_'.join(['rec', m.__name__]): RecScorer(m)})

for model_name in ['shallow', 'deep']:
    X, y, estimator = X_y_model(
        fnames=fnames,
        ages=ages,
        model_name=model_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    scores = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        fit_params={'epochs': n_epochs},
    )
    print(model_name, scores)
