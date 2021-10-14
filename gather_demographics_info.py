import importlib
import pandas as pd

dfs = list()
for dataset in ['chbp', 'lemon', 'tuab', 'camcan']:
    config_map = {'chbp': "config_chbp_eeg",
                  'lemon': "config_lemon_eeg",
                  'tuab': "config_tuab",
                  'camcan': "config_camcan_meg"}
    if dataset not in config_map:
        raise ValueError(f"We don't know the dataset '{dataset}' you requested.")

    cfg = importlib.import_module(config_map[dataset])
    bids_root = cfg.bids_root
    df_sub = pd.read_csv(bids_root / 'participants.tsv', sep='\t')
    df_sub = df_sub[['participant_id', 'age', 'sex']]
    df_sub['dataset'] = dataset
    if 'FEMALE' in df_sub.sex.values:
        df_sub['sex'] = df_sub.sex.map({"FEMALE": "F", "MALE": "M"})
    dfs.append(df_sub)
demog_data = pd.concat(dfs)
demog_data.to_csv('./outputs/demog_summary.csv')

