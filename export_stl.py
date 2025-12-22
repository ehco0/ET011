import openvsp_config # type: ignore
openvsp_config.LOAD_GRAPHICS = False

import openvsp as vsp # type: ignore
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))

def export_stl(
        vsp_file: str,
        stl_file: str = 'outputs/best_wing.stl'
):
    vsp_path = Path(vsp_file)
    stl_path = Path(stl_file)

    stl_path.parent.mkdir(parents=True, exist_ok=True)

    vsp.VSPRenew()
    vsp.ReadVSPFile(str(vsp_path))
    
    compgeom = 'VSPAEROComputeGeometry'
    vsp.SetAnalysisInputDefaults(compgeom)

    vsp.SetIntAnalysisInput(compgeom, 'GeomSet', [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(compgeom, 'ThinGeomSet', [vsp.SET_ALL])

    vsp.ExecAnalysis(compgeom) 
 
    vsp.ExportFile(
        str(stl_path),
        vsp.SET_ALL,
        vsp.EXPORT_STL
    )

    return str(stl_path.resolve())