import cma
import numpy as np
import multiprocessing as mp
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.optimise.objective import objective

def decode(x):
    return {
        'Aspect': 3.0 + x[0] * (25.0 - 3.0),
        'Taper': 0.1 + x[1] * (1.2 - 0.1),
        'Sweep': -10.0 + x[2] * 45.0,
        'Dihedral': -10.0 + x[3] * 10.0,
        'Twist': -2.0 + x[4] * 8.0
    }

def fitness_worker(x, lf_predictor, env, mode):
    x = np.asarray(x, float)

    if np.any(x < 0.0) or np.any(x > 1.0):
        return 1e6

    geom = decode(x)
    aero_sweep = lf_predictor.predict_sweep(
        geom, env['Re'], env['Alpha_sweep']
    )

    geom_metrics = {'AR': geom['Aspect']}
    return objective(aero_sweep, mode=mode, geom_metrics=geom_metrics)

def print_sweep_table(title, aero_sweep, alpha_sweep):
    print(f'\n{title}')
    print(f'{"Alpha":>8s}  {"CL":>10s}  {"CD":>12s}  {"L/D":>10s}')
    print('-' * 46)

    for a, cl, cd, ld in zip(
        alpha_sweep,
        aero_sweep['CL'],
        aero_sweep['CD'],
        aero_sweep['LD']
    ):
        print(f'{float(a):8.3f}  {float(cl):10.5f}  {float(cd):12.6f}  {float(ld):10.5f}')

def optimise_wing(
    lf_predictor,
    env,
    mode='glider',
    maxiter=40,
    popsize=14,
    seed=42,
    n_workers=None
):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    x0 = [0.5] * 5
    sigma0 = 0.6

    init_geom = decode(x0)
    init_aero = lf_predictor.predict_sweep(
        init_geom, env['Re'], env['Alpha_sweep']
    )

    print('Initial geometry:')
    for k, v in init_geom.items():
        print(f'  {k:<10s}: {v:.4f}')

    print_sweep_table(
        'Initial aero sweep',
        init_aero,
        env['Alpha_sweep']
    )

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            'bounds': [[0]*5, [1]*5],
            'maxiter': maxiter,
            'popsize': popsize,
            'seed': seed
        }
    )

    fitness = partial(
        fitness_worker,
        lf_predictor=lf_predictor,
        env=env,
        mode=mode
    )

    with mp.Pool(processes=n_workers) as pool:
        while not es.stop():
            solutions = es.ask()
            fitness_vals = pool.map(fitness, solutions)
            es.tell(solutions, fitness_vals)
            es.disp()

    best_x = es.result.xbest
    best_geom = decode(best_x)
    best_aero = lf_predictor.predict_sweep(
        best_geom, env['Re'], env['Alpha_sweep']
    )

    print('\nBest geometry:')
    for k, v in best_geom.items():
        print(f'  {k:<10s}: {v:.4f}')

    print_sweep_table(
        'Best aero sweep',
        best_aero,
        env['Alpha_sweep']
    )

    return best_geom, best_aero
