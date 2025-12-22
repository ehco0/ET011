from joblib import load, dump
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

from xgboost import XGBRegressor # type: ignore

ROOT = Path(__file__).resolve().parents[2]
HF_CSV = ROOT / 'data' / 'hf_data.csv'
OUT_DIR = ROOT / 'models' / 'correction'
LF_DIR = ROOT / 'models' / 'low fidelity'
OUT_DIR.mkdir(exist_ok=True)

USE_LOGRE = True

BASE_FEATURES = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist', 'Alpha'] # inputs (X)

GROUP_COLS = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist'] # wing geom
# supposed column names for hf_data.csv
CL_HF_COL = "CLtot_HF"
CD_HF_COL = "CDtot_HF" 

# accuracy report 
def report(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'[{name}] MAE={mae:.6g}  RMSE={rmse:.6g}  R2={r2:.6g}')
    return mae, rmse, r2

# grouping ensures each unique wing geom is assigned to either train or test, model is eval on unseen geoms 
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
    hf_df = pd.read_csv(HF_CSV) 
    
    if USE_LOGRE:
        hf_df['logRe'] = np.log10(hf_df['Re'])
        FEATURES = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist', 'Alpha', 'logRe']
    else:
        FEATURES = BASE_FEATURES    

    lf_model_cl = load(LF_DIR / 'xgb_lf_CLtot.joblib')
    lf_model_cd = load(LF_DIR / 'xgb_lf_CDtot.joblib')

    X = hf_df[FEATURES].astype(float)
    CL_lf = lf_model_cl.predict(X)
    CD_lf = lf_model_cd.predict(X)

    # ensure CD positive
    CD_lf = np.clip(CD_lf, 1e-10, None)

    hf_df['CL_LF'] = CL_lf
    hf_df['CD_LF'] = CD_lf

    # correction: change = hf - lf
    hf_df['dCL'] = hf_df[CL_HF_COL].to_numpy() - hf_df['CL_LF'].to_numpy()
    hf_df['dCD'] = hf_df[CD_HF_COL].to_numpy() - hf_df['CD_LF'].to_numpy()  # log CD later maybe --> check whether is useful
    '''
    hf_df['logCD_HF'] = np.log(hf_df[CD_HF_COL])
    hf_df['logCD_LF'] = np.log(CD_lf)
    hf_df['dlogCD'] = hf_df['logCD_HF'] - hf_df['logCD_LF']
    '''
    # corr model inputs now have both X (hf) and lf preds
    CORR_FEATURES = FEATURES + ['CL_LF', 'CD_LF']
    Xcorr = hf_df[CORR_FEATURES].astype(float)

    # split data to 80% train, 20% test by geom groups
    groups = make_groups(hf_df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(Xcorr, hf_df[['dCL', 'dCD']], groups=groups))

    X_train, X_test = Xcorr.iloc[train_idx], Xcorr.iloc[test_idx]
    hf_train, hf_test = hf_df.iloc[train_idx].copy(), hf_df.iloc[test_idx].copy()
    
    print(f'\nRows: train={len(train_idx):,}  test={len(test_idx):,}')
    print(f'Unique geometries: train={groups.iloc[train_idx].nunique():,}  test={groups.iloc[test_idx].nunique():,}')

    corr_cl = make_corr_model(seed=42)
    corr_cd = make_corr_model(seed=43)

    # train correction model by mapping geom + environment (FEATURES) [inputs] to --> lf to hf error correction [outputs]
    corr_cl.fit(X_train, hf_train['dCL'],
                eval_set=[(X_test, hf_test['dCL'])],
                verbose=False)
    
    # see if extra params help cl first ^^^^
    corr_cd.fit(X_train, hf_train['dCD'])

    # predict "lf" data for test inputs
    CL_lf_test = hf_test['CL_LF'].to_numpy()
    CD_lf_test = hf_test['CD_LF'].to_numpy()

    # predict correction change data for test inputs
    dCL_pred = corr_cl.predict(X_test)
    dCD_pred = corr_cd.predict(X_test)

    # predicted corrected coeffs
    CL_pred = CL_lf_test + dCL_pred
    CD_pred = CD_lf_test + dCD_pred

    # actual hf data for same test inputs
    CL_true = hf_test[CL_HF_COL].to_numpy()
    CD_true = hf_test[CD_HF_COL].to_numpy()

    print('\nLow fidelity VS Corrected')
    report(CL_true, CL_lf_test, 'CL hf VS lf')
    report(CL_true, CL_pred, 'CL hf VS corrected')

    report(CD_true, CD_lf_test, 'CD hf VS lf')
    report(CD_true, CD_pred, 'CD hf VS corrected')

    # path to save the correction models for final model
    dump(corr_cl, OUT_DIR / 'corr_dCL.joblib')
    dump(corr_cd, OUT_DIR / 'corr_dCD.joblib')
    print('\nSaved models to:', OUT_DIR.resolve())

if __name__ == '__main__':
    main()

''' wrap for final inverse model --> easier rather than path????
class CorrectedModel:
    def __init__(self, lf_model, corr_model):
        self.lf = lf_model
        self.corr = corr_model

    def predict(self, X):
        lf = self.lf.predict(X)
        return lf + self.corr.predict(X)
'''