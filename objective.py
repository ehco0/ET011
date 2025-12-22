import numpy as np

def objective(aero_sweep, mode='glider', geom_metrics=None):

    CL = np.array(aero_sweep['CL'])
    CD = np.array(aero_sweep['CD'])
    CD = np.clip(CD, 1e-8, None)
    LD = np.array(aero_sweep['LD'])
    Alpha = np.array(aero_sweep['Alpha'])

    mean_LD = np.mean(LD)
    max_LD = np.max(LD)
    mean_CD = np.mean(CD)
    cl_std = np.std(CL)

    dCL = np.diff(CL)
    alpha_mid = Alpha[:-1]

    STALL_ALPHA_MIN = 8.0      
    DCL_THRESH = -0.15          

    early_stall = (alpha_mid < STALL_ALPHA_MIN) & (dCL < DCL_THRESH)
    early_stall_penalty = np.sum(early_stall)

    if mode == 'glider':
        score = (
            + 2.0 * mean_LD
            + 1.0 * max_LD
            - 0.3 * mean_CD
            - 0.5 * cl_std
            - 3.0 * early_stall_penalty
        )
        ar_max = 22.0

    elif mode == 'normal':
        score = (
            + 1.5 * np.mean(CL)
            - 1.0 * mean_CD
            - 0.3 * cl_std
            - 3.5 * early_stall_penalty
        )
        ar_max = 12.0

    else:
        raise ValueError(f'Unknown mode: {mode}')

    penalty = 0.0
    if geom_metrics is not None:
        ar = geom_metrics.get('AR', None)

        def hinge_sq(x):
            return max(0.0, x) ** 2

        if ar is not None:
            penalty += 2000.0 * hinge_sq(ar - ar_max)

    return -float(score) + penalty
