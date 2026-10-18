import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "../data/preprocesaods/merge_clean.csv"
SPLITS_PATH = "../data/preprocesados/cv_splits_10.json"
FEATURE_COLS = [
    "PTGENDER",
    "AGE_T0",
    "PTEDUCAT",
    "APOE4CONT",
    "DIAGNOSIS_T0",
    "DIAGNOSIS_T1",
    "MMSCORE_T0",
    "TOTAL13_T0",
    "CDRSB_T0",
    "MMSCORE_T1",
    "TOTAL13_T1",
    "CDRSB_T1",
    "DELTA_MMSCORE",
    "DELTA_TOTAL13",
    "DELTA_CDRSB",
]
SCORE_COLS = ["MMSCORE_T2", "TOTAL13_T2", "CDRSB_T2"]
DIAG_COLS = "DIAGNOSIS_T2"

df = pd.read_csv(DATASET_PATH)
with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

def load_scaled_i_fold(fold_idx):
    train_rids = splits[fold_idx]["train_rids"]
    val_rids = splits[fold_idx]["val_rids"]
    test_rids = splits[fold_idx]["test_rids"]

    train_mask = df["RID"].isin(train_rids)
    val_mask = df["RID"].isin(val_rids)
    test_mask = df["RID"].isin(test_rids)

    X_train = df.loc[train_mask, FEATURE_COLS].values
    X_val = df.loc[val_mask, FEATURE_COLS].values
    X_test = df.loc[test_mask, FEATURE_COLS].values

    y_scores_train = df.loc[train_mask, SCORE_COLS].values
    y_scores_val = df.loc[val_mask, SCORE_COLS].values
    y_scores_test = df.loc[test_mask, SCORE_COLS].values

    y_diag_train = df.loc[train_mask, DIAG_COLS].values
    y_diag_val = df.loc[val_mask, DIAG_COLS].values
    y_diag_test = df.loc[test_mask, DIAG_COLS].values

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_scores_train": y_scores_train,
        "y_scores_val": y_scores_val,
        "y_scores_test": y_scores_test,
        "y_diag_train": y_diag_train,
        "y_diag_val": y_diag_val,
        "y_diag_test": y_diag_test,
    }