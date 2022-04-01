# M/EEG brain age prediction benchmark paper

This repository presents the code, tools and resources developed in the course of [1]. To reuse the code, please follow the instructions and recommendations below.

[1] D. Engemann, A. Mellot, R. Höchenberger, H. Banville, D. Sabbagh, L. Gemein, T. Ball, and A. Gramfort. A reusable benchmark of brain-age prediction from M/EEG resting-state signals (in preparation).

---

## Exploring the aggregated results using the plotting scripts

For convenience, we provide aggregated group-level results facilitating exploration.

1. Aggregate information on demographics is presented in: ```./outputs/demog_summary_table.csv```
2. Aggregate cross-validation results can be found for every dataset and benchmark in: ```./results/```. Filenames indicate the benchmark and the dataset as in ```./results/benchmark-deep_dataset-lemon.csv``` for the deep learning (Deep4Net) benchmark on the LEMON dataset.

The scripts used for generating the figures and tables presented in the paper can be a good starting point.
All plots and tables were generated using ```plot_benchmark_age_prediction.r```.

The R code uses few dependencies and base-r idioms supporting newer as well as older versions of R.

If needed, dependencies can be installed as follows:

```R
install.packages(c("ggplot2", "scales", "ggThemes", "patchwork", "kableExtra"))
```

The demographic data can be plotted using ```plot_demography.r```. Note however that the input file contains individual-specific data and cannot be readily shared. Computing the input tables can be done using ```gather_demographics_info.py``` provided that all input datasets are correctly downloaded and stored in the BIDS format.

---

## General workflow for computing intermediate outputs

Here we considered 4 datasets that can be programmatically downloaded from their respective websites linked below:

1. [Cam-CAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)
2. [LEMON](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html)
3. [CHBP](https://www.synapse.org/#!Synapse:syn22324937)
4. [TUAB](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuab)

Some of these datasets already come in BIDS formats, others have to be actively converted. In other cases some modifications and fixes are needed to make things work. Please consider the notes on dataset-specific peculiarities below.

Datasets are then preprocessed using the [MNE-BIDS pipeline](https://mne.tools/mne-bids-pipeline/). To make this work, you must edit the respective config files to point to the input and derivative folders on your machine. The respective variables to modify in each config file are ```bids_root``` (input data path), ```deriv_root``` (intermediate BIDS outpouts) and ```subjects_dir``` (FreeSurfer path).

*Note:* The filterbank source model requires on the Cam-Can dataset a reduced size template head model.
After setting your FreeSurfer `subjects_dir` you can obtain the scaled MRI called `fsaverage_small`
using:

```python
mne.coreg.scale_mri("fsaverage", "fsaverage_small", scale=0.9, subjects_dir=subjects_dir, annot=True, overwrite=True)
```

The four config files for the datasets are:

1. ```config_camcan_meg.py```
2. ```config_lemon_eeg.py```
3. ```config_chbp_eeg.py```
4. ```config_tuab_eeg.py```

Once all data is downloaded and the configs are updated, the MNE-BIDS pipeline can be used for preprocessing. We recommend downloading the MNE-BIDS-pipeline repository and placing it in the same folder this repository is downloaded, such that its relative position would be ```../mne-bids-pipeline```. For help with the installation dependencies, please consider the dedicated section below.

*Note:* the MNE-BIDS pipeline is a bit different from other packages. Instead of installing it as a library it is more like a collection of scripts. Installing it means getting the Python files and making sure the dependencies are met. See
[installation instructions](https://mne.tools/mne-bids-pipeline/getting_started/install.html).

If all is good to go, preprocessing can be conducted using the following shell commands:

```bash
python ../mne-bids-pipeline/run.py --config config_camcan_meg.py --n_jobs 40 --steps=preprocessing
python ../mne-bids-pipeline/run.py --config config_lemon_eeg.py --n_jobs 40 --steps=preprocessing
python ../mne-bids-pipeline/run.py --config config_chbp_eeg.py --n_jobs 40 --steps=preprocessing
python ../mne-bids-pipeline/run.py --config config_tuab_eeg.py --n_jobs 40 --steps=preprocessing
```

*Note:* Make sure chose an appropriate number of jobs given your computer. Above 40 are used
but this assumes you have access to a machine with more than 40 CPUs and a lot of RAM.

*Note:* It can be convenient to run these commands from within IPython e.g. to benefit from a nicer Terminal experience during debugging. Start IPython and use ```run``` instead of ```python```.
This will apply filtering and epoching according to the settings in the config files.

Then a custom preprocessing step has to be performed involving artifact rejection and re-referencing:

```bash
python compute_autoreject.py --n_jobs 40
```

*Note:* This will run computation for all datasets. To perform this step on specific datasets, check out the `-d` argument.

Once this is done, some additional processing steps are needed for the filterbank-source models:

```bash
python ../mne-bids-pipeline/run.py --config config_camcan_meg.py --n_jobs 40 --steps=source
python ../mne-bids-pipeline/run.py --config config_lemon_eeg.py --n_jobs 40 --steps=source
python ../mne-bids-pipeline/run.py --config config_chbp_eeg.py --n_jobs 40 --steps=source
python ../mne-bids-pipeline/run.py --config config_tuab_eeg.py --n_jobs 40 --steps=source
```

Potential errors can be inspected in the ```autoreject_log.csv``` that is written in the dataset-specific derivative directories.

Now feature computation can be launched for the 3 non-deep learning benchmarks:

1. handcrafted
2. filterbank-riemann
3. filterbank-source

```bash
python compute_features.py --n_jobs 40
```

*Note:* This will run computation for all datasets and all benchmarks. To visit specific datasets or feature types, check out the `-d` and `-f` arguments.

Potential errors can be inspected in the benchmark-specific log files that are written in the dataset-specific derivative directories, e.g. ```feature_fb_covs_pooled-log.csv``` for the filterbank features.

If all went fine until now, the following machine learning benchmarks can finally be run:

0. dummy
1. handcrafted
2. filterbank-riemann
3. filterbank-source
4. shallow
5. deep

```bash
python compute_benchmark_age_prediction.py --n_jobs 10
```

*Note:* This will run computation for all datasets and all benchmarks. To visit specific datasets or benchmarks, checkout the `-d` and `-b` arguments.

If all worked until now out you should find the fold-wise scores for every benchmark on every dataset in ```./results```.

---

## Handling dataset-specific peculiarities prior to computation

For some of the datasets, custom processing of the input data was necessary.

#### Cam-CAN

The dataset was provided in BIDS format by the curators of the Cam-CAN.
Computation worked out of the box.
Note that previous releases were not provided in BIDS format.
Moreover, Maxwell filtering was applied for mitigating strong environmental magnetic artifacts which is only available for MEG and not EEG.

#### LEMON

##### Downloading the data

The data provided by LEMON can be conveniently downloaded using our custom script:

```download_data_lemon.py```

Make sure to adjust the paths to make this work on your machine.
Also note that the script presupposes that the ```ETA_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv``` file has been dowloaded to this repository.

##### Finishing BIDS conversion

Further steps are necessary to obtain a fully operable BIDS dataset.
That effort is summarized in ```convert_lemon_to_bids.py```.

##### Manual fixes

We noticed that for the following subjects the header files were pointing to
data files with an old naming scheme, leading to errors upon file-reading:

- sub-010193
- sub-010044
- sub-010219
- sub-010020
- sub-010203

For these subjects, we had to manually edit the `*.vhdr` files to point to the bids name of the marker and data files, e.g. `sub-010193.eeg`, `sub-010193.vmrk`.
This error may be fixed in a future release of the LEMON data.

#### CHBP

##### Downloading the data

The data can be downloaded from synapse: https://doi.org/10.7303/syn22324937

It can be handy to use the command-line client for programmatic download, which can be installed using pip:

```bash
pip install synapseclient
```

Then one can log in using one's credentials ...

```bash
synapse login -u "username" -p "password"
```

... and download specific folders recursively:

```bash
synapse get -r syn22324937
```

##### Finishing BIDS conversion

Further steps were needed to make the CHBP data work using the MNE-BIDS package.
That effort is summarized in ```convert_chbp_to_bids.py```.

Note that future versions of the dataset may require modifications to this approach or render some of these measures unnecessary.

The current work is based on the dataset as it was available in July 2021.

##### Manual fixes

We found a bug in the `participants.tsv` file, leading to issues with the BIDS validator.
In the input data (July 2021), one can find a trailing whitespace until line 251. Then the line terminates at the last character of the “sex” column (F/M). We removed the whitespaces to ensure proper file-parsing.

Note that future versions of the dataset may require modifications to this approach or render some of these measures unnecessary.

#### TUAB

##### BIDS conversion

After downloading the TUAB data, we first needed to create a BIDS dataset.
That effort is summarized in ```convert_tuh_to_bids.py```.

---

### Installation of packages and dependencies

The development initiated by this work has been stabilized and released in the latest versions of packages we list as dependencies below. You can install these packages using pip. For their respective dependencies, consider the package websites:

1. [MNE](https://mne.tools/stable/install/index.html)

2. [MNE-bids](https://mne.tools/mne-bids/v0.3/index.html)

4. [AutoReject](https://autoreject.github.io/stable/index.html)

5. [coffeine](https://github.com/coffeine-labs/coffeine)

6. [Braindecode](https://github.com/braindecode/braindecode)

The MNE-BIDS pipeline repository is not a package in the classical sense. We recommend using the latest version from GitHub. Please consider the [installation instructions](https://mne.tools/mne-bids-pipeline/getting_started/install.html).
