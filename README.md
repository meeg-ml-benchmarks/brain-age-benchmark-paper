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

### General worklfow

Here we considered 4 datasets that can be programmatically downloaded from their respective websites linked below:

1. [Cam-CAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)
2. [LEMON](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html)
3. [CHBP](https://www.synapse.org/#!Synapse:syn22324937)
4. [TUAB](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuab)

Some of these datasets already come in BIDS formats, others have to be actively converted. In other cases some modifications and fixes are needed to make things work. Please consider the notes on dataset-specific peculiarities.

Datsets are then preprocessed using the MNE-BIDS pipeline. To make this work, you must edit the respective config files
to point to the input and derivative folders on your machine. The respective variables to modify in each config file are ```bids_root``` (input data path), ```deriv_root``` (intermediate BIDS outpouts) and ```subjects_dir``` (freesurfer path).

The four config files for the datasets are:

1. ```config_camcan_meg.py```
2. ```config_lemon_eeg.py```
3. ```config_chbp_eeg.py```
4. ```config_tuab_eeg.py```

Once all data is downloaded and the configs are updated, the MNE-BIDS pipeline can be used for preprocessing. We recommend downloading the MNE-BIDS-pipeline repository and placing it in the same folder this repository is downloaded, sucht that its releative position would be ```../mne-bids-pipeline```. For help with the installation dependencies, please consider the dedicated section below. 

*Note:* the bids pipeline is a bit different from other packages. Instead of installing it as a library it is more like a collection of scripts. Installing it means cloning the GitHub repository and making sure the dependencies are met.

If all is good to go, preprocessing can be conducted using the following shell commands.

```bash
python ../mne-bids-pipeline/run.py --config config_camcan_meg.py --n_jobs 40 --steps=preprocessing
python ../mne-bids-pipeline/run.py --config config_lemon_eeg.py --n_jobs 40 --steps=preprocessing
python ../mne-bids-pipeline/run.py --config config_chbp_eeg.py --n_jobs 40 --steps=preprocessing
python ../mne-bids-pipeline/run.py --config config_tuab_eeg.py --n_jobs 40 --steps=preprocessing
```

*Note:* Make sure chose an appropriate number of jobs given your computer.

*Note:* It can be convenient to run these commans from within IPython e.g. to benefit from a nicer Terminal experience during debugging. Start IPython and use ```run``` instead of ```python```

This will apply filtering and epoching according to the settings in the config files.

Then a custom preprocessing step has to be performed involving artifact rejection and re-referencing.

```bash
python compute_autoreject.py --n_jobs 40
```

*Note:* This will run computation for all datasets. To visit specific datasets, checkout the `-d` argument.

Once this is done, the additional processing needed for the filterbank-source models has to be conducted:

```bash
python ../mne-bids-pipeline/run.py --config config_camcan_meg.py --n_jobs 40 --steps=source
python ../mne-bids-pipeline/run.py --config config_lemon_eeg.py --n_jobs 40 --steps=source
python ../mne-bids-pipeline/run.py --config config_chbp_eeg.py --n_jobs 40 --steps=source
python ../mne-bids-pipeline/run.py --config config_tuab_eeg.py --n_jobs 40 --steps=source
```

Potential errors can be inpsected in the ```autoreject_log.csv``` that is written in the dataset-specific derivative directories.

Now feature computation can be launched for the 3 non-deep learning benchmarks:

1. handcrafted
2. filterbank-riemann
3. filterbank-source

```bash
python compute_features.py --n_jobs 40
```
*Note:* This will run computation for all datasets and all benchmarks. To visit specific datasets or feature types, checkout the `-d` and `-f` arguments.

Potential errors can be inpsected in the benchmark-specific logfiles that is written in the dataset-specific derivative directories, e.g. ```feature_fb_covs_pooled-log.csv``` for the filterbank features.

If all went fine until now, the machine learning benchmarks:

0. dummy
1. handcrafted
2. filterbank-riemann
3. filterbank-source
4. shallow
5. deep

```bash
python compute_benchmark_age_prediction.py --n_jobs 10
```

*Note:* This will run computation for all datasets and all benchmarks. To visit specific datasets or benchnmarks, checkout the `-d` and `-b` arguments.

If all worked until now out you should find the fold-wise scores for every benchmark on every dataset in ```./results``` 

### Datset-specific peculiarities

### Installation of packages and dependencies

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
