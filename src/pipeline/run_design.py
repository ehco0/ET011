from pathlib import Path
import sys
import numpy as np
import torch
import joblib
import pickle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.surrogate.nn_surrogate import NNSurrogate
from src.optimise.cmaes_optimise import optimise_wing_nn
from src.vsp.build_wing import build_wing
from src.vsp.export_stl import export_stl
from src.surrogate.model_def import AeroMultiNN

print('OK')

def run_design(
    Re: float,
    Alpha_start: float,
    Alpha_end: float,
    n_alpha: int = 16,
    mode: str = 'glider',
    out_dir = 'outputs'
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AeroMultiNN(input_dim=9)
    model.load_state_dict(
        torch.load(ROOT / 'models' / 'trained_model.pt', map_location=device)
    )

    with open(ROOT / 'models' / 'norm_stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    xgb_CL = joblib.load(ROOT / 'models' / 'low fidelity' / 'lf_CLtot.joblib')
    xgb_CD = joblib.load(ROOT / 'models' / 'low fidelity' / 'lf_CDtot.joblib')

    surrogate = NNSurrogate(
        model=model,
        CL_mean=stats['CL_mean'], CL_std=stats['CL_std'],
        dlogCD_mean=stats['dlogCD_mean'], dlogCD_std=stats['dlogCD_std'],
        LD_mean=stats['LD_mean'], LD_std=stats['LD_std'],
        xgb_CL=xgb_CL,
        xgb_CD=xgb_CD,
        device=device
    )

    env = {
        'Re': float(Re),
        'Alpha_sweep': np.linspace(Alpha_start, Alpha_end, n_alpha)
    }

    best_geom, best_aero = optimise_wing_nn(
        surrogate,
        env,
        mode=mode,
        maxiter=60,
        popsize=20,
        seed=42
    )

    vsp_file = str(out_dir / 'best_wing.vsp3')
    stl_file = str(out_dir / 'best_wing.stl')

    vsp_path = build_wing(best_geom, outfile=vsp_file)
    stl_path = export_stl(vsp_path, stl_file=stl_file)

    print('\nInput environment:')
    print(f'{'Re':11s}: {env['Re']}')
    print(f'{'Alpha Start':11s}: {Alpha_start}')
    print(f'{'Alpha End':11s}: {Alpha_end}')
    print(f'{'Mode':11s}: {mode}')

    print('\nBest geometry:')
    for k, v in best_geom.items():
        print(f'{k:10s}: {v:.4f}')

    for a, cl, cd, ld in zip(
        env['Alpha_sweep'],
        best_aero['CL'],
        best_aero['CD'],
        best_aero['LD']
    ):
        print(f'{a:8.3f}  {cl:10.5f}  {cd:12.6f}  {ld:10.5f}')

    return {
        'env': env,
        'best_geom': best_geom,
        'best_aero': best_aero,
        'vsp_file': vsp_path,
        'stl_file': stl_path
    }


if __name__ == '__main__':
    run_design(
        Re=1e7,
        Alpha_start=-10.0,
        Alpha_end=20.0,
        mode='normal'
    )

