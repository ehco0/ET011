import numpy as np
import torch
import joblib
import pickle
import cma

from pathlib import Path

from nn_surrogate import NNSurrogate
from objective import objective
from model_def import AeroMultiNN   

ROOT = Path(__file__).resolve().parents[2]

RE = 2.0e6
ALPHA_SWEEP = np.linspace(-4.0, 14.0, 25)
MODE = "glider"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AeroMultiNN(input_dim=9)
model.load_state_dict(
    torch.load(ROOT / "models" / "trained_model.pt", map_location=device)
)

with open(ROOT / "models" / "norm_stats.pkl", "rb") as f:
    stats = pickle.load(f)

xgb_CL = joblib.load(ROOT / "models" / "low fidelity" / "lf_CLtot.joblib")
xgb_CD = joblib.load(ROOT / "models" / "low fidelity" / "lf_CDtot.joblib")

surrogate = NNSurrogate(
    model=model,
    CL_mean=stats["CL_mean"], CL_std=stats["CL_std"],
    dlogCD_mean=stats["dlogCD_mean"], dlogCD_std=stats["dlogCD_std"],
    LD_mean=stats["LD_mean"], LD_std=stats["LD_std"],
    xgb_CL=xgb_CL,
    xgb_CD=xgb_CD,
    device=device
)

def evaluate_design(x):

    geom = {
        "Aspect":   x[0],
        "Taper":    x[1],
        "Sweep":    x[2],
        "Dihedral": x[3],
        "Twist":    x[4],
    }

    aero = surrogate.predict_sweep(
        geom_dict=geom,
        Re=RE,
        alpha_sweep=ALPHA_SWEEP
    )

    # objective REQUIRES Alpha inside aero_sweep
    aero["Alpha"] = ALPHA_SWEEP

    geom_metrics = {
        "AR": geom["Aspect"]
    }

    return objective(
        aero_sweep=aero,
        mode=MODE,
        geom_metrics=geom_metrics
    )

x0 = np.array([
    12.0,   # Aspect ratio
    0.45,   # Taper
    15.0,   # Sweep (deg)
    4.0,    # Dihedral (deg)
    -2.0    # Twist (deg)
])

sigma0 = 0.4

bounds = [
    [6.0,  0.2,  0.0,  0.0,  -6.0],   # lower
    [22.0, 0.8, 35.0, 10.0,  2.0],    # upper
]

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "bounds": bounds,
        "popsize": 12,
        "maxiter": 120,
        "seed": 42,
        "verb_disp": 1,
    }
)

while not es.stop():
    solutions = es.ask()
    values = [evaluate_design(x) for x in solutions]
    es.tell(solutions, values)
    es.disp()

res = es.result

print("\n==============================")
print("Best design found")
print("==============================")
print("x =", res.xbest)
print("objective =", res.fbest)

np.save(ROOT / "models" / "best_design.npy", res.xbest)
print("Saved best design to models/best_design.npy")
