from joblib import load, dump
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
PAIRED_CSV = ROOT / 'data' / 'paired_data.csv'
OUT_DIR = ROOT / 'models' / 'correction'
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LOGRE = True

BASE_FEATURES = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist', 'Alpha']

GROUP_COLS = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist'] 
# column names for paired_data.csv
CL_LF_COL = 'CLtot_LF'
CD_LF_COL = 'CDtot_LF'
CL_HF_COL = 'CLtot_HF'
CD_HF_COL = 'CDtot_HF'

def report(y_true, y_pred, name: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'[{name}] MAE={mae:.6g}  RMSE={rmse:.6g}  R2={r2:.6g}')
    return mae, rmse, r2

def make_groups(df):
    return df[GROUP_COLS].round(6).astype(str).agg('|'.join, axis=1)

def make_corr_model(seed=42):
    return XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=5.0,
        objective='reg:squarederror',
        random_state=seed,
        n_jobs=-1,
    )

def main():
    df = pd.read_csv(PAIRED_CSV)

    if USE_LOGRE:
        df['logRe'] = np.log10(df['Re'])
        FEATURES = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist', 'Alpha', 'logRe']
    else:
        FEATURES = BASE_FEATURES

    df['dCL'] = df[CL_HF_COL].to_numpy() - df[CL_LF_COL].to_numpy()
    df['dCD'] = df[CD_HF_COL].to_numpy() - df[CD_LF_COL].to_numpy()
   
    CORR_FEATURES = BASE_FEATURES + [CL_LF_COL, CD_LF_COL]
    X = df[CORR_FEATURES].astype(float)

    # split data
    groups = make_groups(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, df[['dCL', 'dCD']], groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    df_train, df_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    print(f'Split: train={len(train_idx):,}  test={len(test_idx):,}')
    print(f'Unique geometries: train={groups.iloc[train_idx].nunique():,}  test={groups.iloc[test_idx].nunique():,}')

    corr_cl = make_corr_model(seed=42)
    corr_cd = make_corr_model(seed=43)
 
    corr_cl.fit(X_train, df_train['dCL'])
    corr_cd.fit(X_train, df_train['dCD'])

    CL_hf = df_test[CL_HF_COL].to_numpy()
    CD_hf = df_test[CD_HF_COL].to_numpy()

    CL_lf = df_test[CL_LF_COL].to_numpy()
    CD_lf = df_test[CD_LF_COL].to_numpy()

    dCL_pred = corr_cl.predict(X_test)
    dCD_pred = corr_cd.predict(X_test)

    CL_pred = CL_lf + dCL_pred
    CD_pred = CD_lf + dCD_pred

    print('Low fidelity VS corrected')
    report(CL_hf, CL_lf,   'CL hf vs lf')
    report(CL_hf, CL_pred, 'CL hf vs corrected')

    report(CD_hf, CD_lf,   'CD hf vs lf')
    report(CD_hf, CD_pred, 'CD hf vs corrected')

    dump(corr_cl, OUT_DIR / 'corr_paired_dCL.joblib')
    dump(corr_cd, OUT_DIR / 'corr_paired_dlogCD.joblib')
    print('\nSaved correction models to:', OUT_DIR.resolve())

if __name__ == '__main__':
    main()
