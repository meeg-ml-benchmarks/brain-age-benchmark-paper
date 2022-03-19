# %% imports
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATASETS = ['chbp', 'lemon', 'tuab', 'camcan']
BENCHMARKS = ['dummy', 'filterbank-riemann', 'filterbank-source',
              'handcrafted', 'shallow', 'deep']

parser = argparse.ArgumentParser(description='Compute features.')
parser.add_argument(
    '-d', '--dataset',
    default=None,
    nargs='+',
    help='the dataset for which features should be computed')
parser.add_argument(
    '-b', '--benchmark',
    default=None,
    nargs='+', help='Type of features to compute')

parsed = parser.parse_args()
datasets = parsed.dataset
benchmarks = parsed.benchmark
if datasets is None:
    datasets = DATASETS
if benchmarks is None:
    benchmarks = BENCHMARKS

tasks = [(ds, bs) for ds in datasets for bs in benchmarks]

for dataset, benchmark in tasks:
    print(f"Plotting for '{benchmark}' on '{dataset}' data")
    ys = pd.read_csv(
        f"./results/benchmark-{benchmark}_dataset-{dataset}_ys.csv")
    sns.scatterplot(x="y_true", y="y_pred", data=ys)

plt.show()
