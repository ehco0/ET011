import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump # type: ignore
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error,     r2_score # type: ignore

from xgboost import XGBRegressor # type: ignore
from xgboost.callback import EarlyStopping

ROOT = Path(__file__).resolve().parents[2]
LF_CSV = ROOT / 'data' / 'lf_data.csv'
OUT_DIR = ROOT / 'models' / 'low fidelity'
OUT_DIR.mkdir(exist_ok=True)

USE_LOGRE = True     
CL_MIN, CL_MAX = -10,10
CD_MIN, CD_MAX = 0, 0.5
LD_MIN, LD_MAX = -100, 100

BASE_FEATURES = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist', 'Alpha']
TARGETS = ['CLtot', 'CDtot', 'L_D', 'CMytot']
GROUP_COLS = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist']

def report(y_true, y_pred, name: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'[{name}] MAE={mae:.6g}  RMSE={rmse:.6g}  R2={r2:.6g}')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def make_groups(df):
    return df[GROUP_COLS].round(6).astype(str).agg('|'.join, axis=1)

def make_xgb(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.75, 
        colsample_bytree=0.75,
        reg_alpha=1e-3,
        reg_lambda=5.0,
        min_child_weight=1.0,
        gamma=0.05,
        objective='reg:squarederror',
        random_state=seed,
        early_stopping_rounds=100,
        n_jobs=-1,
        eval_metric='rmse'
    )

def main():
    lf_df = pd.read_csv(LF_CSV)
    print('Raw CDtot max:', lf_df['CDtot'].max(), 
          'Raw CLtot min/max:', lf_df['CLtot'].min(), lf_df['CLtot'].max(),
          'Raw L_D min/max:', lf_df['L_D'].min(), lf_df['L_D'].max(),
          'Raw CMytot min/max:', lf_df['CMytot'].min(), lf_df['CMytot'].max(),)
    
    print('Raw Re unique:', lf_df['Re'].nunique(), 'min/max:', lf_df['Re'].min(), lf_df['Re'].max())

    lf_df = lf_df.dropna(subset=BASE_FEATURES + ['Re'] + TARGETS).copy()
    lf_df = lf_df[lf_df['CDtot'].between(CD_MIN, CD_MAX)].copy()
    lf_df = lf_df[lf_df['CLtot'].between(CL_MIN, CL_MAX)].copy()
    lf_df = lf_df[lf_df['L_D'].between(LD_MIN, LD_MAX)].copy()
    lf_df = lf_df[lf_df['Re'] > 0].copy()

    if USE_LOGRE:
        lf_df['logRe'] = np.log10(lf_df['Re'])
        FEATURES = ['Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist', 'Alpha', 'logRe']
    else:
        FEATURES = BASE_FEATURES

    print('\nAfter cleaning rows:', len(lf_df))
    print('After cleaning CDtot max:', lf_df['CDtot'].max(), 
          'CLtot min/max:', lf_df['CLtot'].min(), lf_df['CLtot'].max(),
          'L_D min/max:', lf_df['L_D'].min(), lf_df['L_D'].max(),
          'CMytot min/max:', lf_df['CMytot'].min(), lf_df['CMytot'].max())
    print('Using FEATURES:', FEATURES)

    groups = make_groups(lf_df)
    X = lf_df[FEATURES].astype(float)
    Y = lf_df[TARGETS].astype(float)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(splitter.split(X, Y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    print(f'\nRows: train={len(X_train):,}  test={len(X_test):,}')
    print(f'Unique geometries: train={groups.iloc[train_idx].nunique():,}  test={groups.iloc[test_idx].nunique():,}')

    models = {}
    for tgt in TARGETS:
        print(f'\nTraining target: {tgt}')
        model = make_xgb(seed=42)
        model.fit(X_train, 
                  Y_train[tgt],
                  eval_set=[(X_train, Y_train[tgt]), (X_test, Y_test[tgt])],
                  verbose=False,
                  )
        pred = model.predict(X_test)
        report(Y_test[tgt], pred, f'TEST {tgt}')
        models[tgt] = model
        dump(model, OUT_DIR / f'lf_{tgt}.joblib')

        # training history
        results = model.evals_result()

        train_rmse = results['validation_0']['rmse']
        test_rmse  = results['validation_1']['rmse']
        best_iter = model.best_iteration

        plt.figure()
        plt.plot(train_rmse, label='Train RMSE')
        plt.plot(test_rmse, label='Validation RMSE')
        plt.axvline(best_iter, linestyle='--', label='Best iteration')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (RMSE)')
        plt.legend()
        plt.title(f'Training history: {tgt}')
        plt.show()

    print(f'\nSaved models to: {OUT_DIR.resolve()}')

if __name__ == '__main__':
    main()
