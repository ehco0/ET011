import openvsp_config # type: ignore
openvsp_config.LOAD_GRAPHICS = False

import openvsp as vsp  # type: ignore
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))

def build_wing(
        geom: dict,
        area: float =  2094.26023,
        outfile: str = 'outputs/best_wing.vsp3'
):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    vsp.VSPRenew()
    wing_id = vsp.AddGeom('WING')

    vsp.SetParmVal(wing_id, 'Aspect', 'XSec_1', geom['Aspect'])
    vsp.SetParmVal(wing_id, 'Taper',  'XSec_1', geom['Taper'])
    vsp.SetParmVal(wing_id, 'Sweep', 'XSec_1', geom['Sweep'])
    vsp.SetParmVal(wing_id, 'Dihedral', 'XSec_1', geom['Dihedral'])
    vsp.SetParmVal(wing_id, 'Twist', 'XSec_1', geom['Twist'])

    vsp.SetParmVal(wing_id, 'TotalArea', 'WingGeom', area)

    vsp.Update()

    vsp.WriteVSPFile(str(outfile))
    return str(outfile.resolve())

