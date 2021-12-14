# M/EEG brain age prediction benchmark paper

This is repository presents the code, the tools and resources developed in the course of [1]. To reuse the code, please follow the instructions and recommendations below.

[1] D. Engemann, A. Mellot, R. Höchenberger, H. Banville, D. Sabbagh, L. Gemein, T. Ball, and A. Gramfort.
(in preparation). 

## Exploring the aggregated results using the plotting scripts

For convenience, we provide aggregated group-level results facilitating exploration.

1. Aggregate information on demographics is presented in: ```./outputs/demog_summary_table.csv```
2. Aggregate cross-validation results can be found for every dataset and benchmark in: ```./results/```. Filenames indicate the benchmark and the dataset as in ```./results/benchmark-deep_dataset-lemon.csv``` for the deep-learning benchmark on the LEMON dataset.

The scripts used for generating the figures and tables presented in the paper can be a good starting point.
All plots and tables were generated using ```plot_benchmark_age_prediction.r```.

The R code is using few dependencies and base-r idioms supporting newer as well as older versions of R.

If needed, dendencies can be installed as follows:

```R
install.packages(c("ggplot2", "scales", "ggThemes", "patchwork", "kableExtra"))
```

The demographic data can be plotted using ```plot_demography.r```. Note however that the input file contains individual-specific data and cannot be readily shared. Computing the input tables can be done using ```gather_demographics_info.py``` provided that all input datasets are correctly downloaded and stored in the BIDS format. 

## Computing the intermediate outputs

To run the code please consider the following instructions.

### Dependencies

1. MNE stable: https://mne.tools/stable/install/index.html

2. [coffeine](https://github.com/dengemann/coffeine) + dependencies: <!-- XXX : pip install coffeine is enough no? -->

    a. https://github.com/dengemann/coffeine#installation-of-python-package

    b. https://pyriemann.readthedocs.io/en/latest/installing.html

    c. https://scikit-learn.org/stable/install.html

3. MNE-bids: https://mne.tools/mne-bids/v0.3/index.html

4. MNE-bids-pipeline: https://mne.tools/mne-bids-pipeline/getting_started/install.html

5. [Braindecode](https://github.com/braindecode/braindecode) and [pytorch](http://pytorch.org/) for deep learning benchmarks

*Note:* the bids pipeline is a bit different from other packages. Instead of installing it as a library it is more like a collection of scripts. Installing it means cloning the GitHub repository and making sure the dependencies are met (link above).

### Running the code

Assuming that `mne-bids-pipeline` is downloaded in the parent directory of this repository you can kick-off the preprocessing like:

```bash
python ../mne-bids-pipeline/run.py --config config_chbp_eeg.py --steps=preprocessing
```

Then run `compute_autoreject.py`, `compute_features.py` and `compute_brain_age.py`.
