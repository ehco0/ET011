import numpy as np
from joblib import load
from pathlib import Path
import sys

class LFPredictor:
    def __init__(self, model_dir):
        ROOT = Path(__file__).resolve().parents[2]
        model_dir = Path(ROOT / model_dir)

        self._model_cl = load(model_dir / 'lf_CLtot.joblib')
        self._model_cd = load(model_dir / 'lf_CDtot.joblib')
        self._model_ld = load(model_dir / 'lf_L_D.joblib' )

        self._features = [
            'Aspect', 'Taper', 'Sweep', 'Dihedral', 'Twist', 'Alpha', 'logRe'
        ]

    def predict_sweep(self, geom, Re, alphas):
        logRe = np.log10(max(Re, 1e-12))
        alphas = np.asarray(alphas, float)

        X = np.column_stack([
            np.full_like(alphas, geom['Aspect'], dtype=float),
            np.full_like(alphas, geom['Taper'], dtype=float),
            np.full_like(alphas, geom['Sweep'], dtype=float),
            np.full_like(alphas, geom['Dihedral'], dtype=float),
            np.full_like(alphas, geom['Twist'], dtype=float),
            alphas,
            np.full_like(alphas, logRe, dtype=float),
        ])

        CL = self._model_cl.predict(X)
        CD = self._model_cd.predict(X)
        CD = np.clip(CD, 1e-8, None)
        LD = self._model_ld.predict(X)

        return {'CL': CL.tolist(), 'CD': CD.tolist(), 'LD': LD.tolist(), 'Alpha': alphas.tolist()}