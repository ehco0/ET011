from joblib import load
import numpy as np
import pandas as pd
from pathlib import Path

class HFPredictor:
    def __init__(self, lf_dir, corr_dir):
        lf_dir = Path(lf_dir)
        corr_dir = Path(corr_dir)

        self.lf_cl = load(lf_dir / 'lf_CLtot.joblib')
        self.lf_cd = load(lf_dir / 'lf_CDtot.joblib')

        self.corr_dcl = load(corr_dir / 'corr_dCL.joblib')
        self.corr_dlogcd = load(corr_dir / 'corr_dCD.joblib')

        self.base_feats = ['Aspect','Taper','Sweep','Dihedral','Twist','Alpha','logRe']
        self.corr_feats = self.base_feats + ['CL_LF','CD_LF']

    def predict_df(self, df: pd.DataFrame):
        df = df.copy()
        df['logRe'] = np.log10(df['Re'].clip(lower=1e-12))

        X_base = df[self.base_feats].astype(float)
        cl_lf = self.lf_cl.predict(X_base)
        cd_lf = np.clip(self.lf_cd.predict(X_base), 1e-10, None)

        df['CL_LF'] = cl_lf
        df['CD_LF'] = cd_lf

        X_corr = df[self.corr_feats].astype(float)
        dcl = self.corr_dcl.predict(X_corr)
        dlogcd = self.corr_dlogcd.predict(X_corr)

        cl = cl_lf + dcl
        cd = np.exp(np.log(cd_lf) + dlogcd)
        ld = cl / cd

        out = df.copy()
        out['CL_pred'] = cl
        out['CD_pred'] = cd
        out['LD_pred'] = ld
        return out
