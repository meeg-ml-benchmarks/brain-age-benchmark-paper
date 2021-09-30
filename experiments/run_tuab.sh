##
# Train brain age models on TUAB
##

# # 1. Prepare TUAB in BIDS format
# python convert_tuh_to_bids.py --healthy_only True --reset_session_indices True

# # 2. Preprocess data
# python ../mne-bids-pipeline/run.py config_tuab.py --steps=preprocessing

# # 2. Compute autoreject
# python compute_autoreject.py --dataset tuab

# # 3. Compute features
# python compute_features.py --dataset tuab

# 4. Train and evaluate models
python compute_brain_age.py --dataset tuab
