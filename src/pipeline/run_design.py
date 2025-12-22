from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))
from src.surrogate.lf_predictor import LFPredictor
from src.optimise.cmaes_optimise import optimise_wing
from src.vsp.build_wing import build_wing
from src.vsp.export_stl import export_stl

print('OK') # check if model can be imported

def run_design(Re: float, 
               Alpha_start: float, 
               Alpha_end: float, 
               n_alpha: int=16,
               mode: str='glider', 
               n_workers=8,
               model_dir = 'models/low fidelity', 
               out_dir = 'outputs'):
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lf = LFPredictor(model_dir)

    env  = {'Re': float(Re), 'Alpha_sweep': np.linspace(Alpha_start, Alpha_end, n_alpha)}

    best_geom, best_aero = optimise_wing(lf, env, mode=mode, maxiter=60, popsize=20, seed=42)

    vsp_file = str((out_dir / 'best_wing.vsp3'))
    stl_file = str((out_dir / 'best_wing.stl'))

    vsp_path = build_wing(best_geom, outfile=vsp_file)
    stl_path = export_stl(vsp_path, stl_file=stl_file)

    print('\nInput environment:')
    print(f'{"Re":11s}: {env["Re"]}')
    print(f'{"Alpha Start":11s}: {Alpha_start}')
    print(f'{"Alpha End":11s}: {Alpha_end}')
    print(f'{"Mode":11s}: {mode}')
    print(f'{"Sweep":11s}: {env["Alpha_sweep"]}')

    print('\nBest geometry:')
    for h, k in best_geom.items():
        print(f'{h:10s}: {k:.4f}')

    for alpha, cl, cd, ld in zip(env['Alpha_sweep'], best_aero['CL'], best_aero['CD'], best_aero['LD']):
        print(f'{alpha:8.3f}  {cl:10.5f}  {cd:12.6f}  {ld:10.5f}')

    return {
        'env': env,
        'best_geom': best_geom,
        'best_aero': best_aero,
        'vsp_file': vsp_path,
        'stl_file': stl_path
    }

if __name__ == '__main__':
    run_design(Re=1e7, Alpha_start=-10.0, Alpha_end=20.0,mode='normal', n_workers=8)
