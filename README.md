# EEG predictive modeling summerschool
Age prediction benchmark project


## Notebooks

Check out:

1. `plot_eeg_features_and_build_a_simple_model.ipynb`

## Computing the intermediate outputs

To run the code please consider the following instructions.

### Dependencies

1. MNE stable: https://mne.tools/stable/install/index.html
	
2. coffeine + dependencies: 

    a. https://github.com/dengemann/coffeine#installation-of-python-package

    b. https://pyriemann.readthedocs.io/en/latest/installing.html

    c. https://scikit-learn.org/stable/install.html

3. MNE-bids: https://mne.tools/mne-bids/v0.3/index.html
	
4. MNE-bids-pipeline: https://mne.tools/mne-bids-pipeline/getting_started/install.html 
  
*Note*: the bids pipeline is a bit different from other packages. Instead of installing it as a library it is more like a collection of scripts. Installing it means cloning the GitHub repository and making sure the dependencies are met (link above).

### Running the code

Assuming that `mne-bids-pipeline` is downloaded in the parent directory of this repository you can kick-off the preprocessing like:

```bash
python ../mne-bids-pipeline/run.py  config_chbp_eeg.py --steps=preprocessing
```

Then run `compute_autoreject.py` and `compute_features.py`

