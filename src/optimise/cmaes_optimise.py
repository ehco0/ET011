import numpy as np
import cma
from src.optimise.objective import objective


def optimise_wing_nn(
    surrogate,
    env,
    mode='glider',
    maxiter=60,
    popsize=20,
    seed=42
):

    Re = env['Re']
    Alpha_sweep = env['Alpha_sweep']

    def evaluate(x):

        geom = {
            'Aspect':   x[0],
            'Taper':    x[1],
            'Sweep':    x[2],
            'Dihedral': x[3],
            'Twist':    x[4],
        }

        aero = surrogate.predict_sweep(
            geom_dict=geom,
            Re=Re,
            alpha_sweep=Alpha_sweep
        )

        aero['Alpha'] = Alpha_sweep

        geom_metrics = {
            'AR': geom['Aspect']
        }

        return objective(
            aero_sweep=aero,
            mode=mode,
            geom_metrics=geom_metrics
        )

    x0 = np.array([12.0, 0.45, 15.0, 4.0, -2.0])
    sigma0 = 0.4

    bounds = [
        [6.0,  0.2,  0.0,  0.0,  -6.0],
        [22.0, 0.8, 35.0, 10.0,  2.0],
    ]

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            'bounds': bounds,
            'popsize': popsize,
            'maxiter': maxiter,
            'seed': seed,
            'verb_disp': 1,
        }
    )
    
    while not es.stop():
        X = es.ask()
        f = [evaluate(x) for x in X]
        es.tell(X, f)
        es.disp()

    res = es.result
    xbest = res.xbest

    best_geom = {
        'Aspect':   xbest[0],
        'Taper':    xbest[1],
        'Sweep':    xbest[2],
        'Dihedral': xbest[3],
        'Twist':    xbest[4],
    }

    best_aero = surrogate.predict_sweep(
        geom_dict=best_geom,
        Re=Re,
        alpha_sweep=Alpha_sweep
    )

    return best_geom, best_aero
