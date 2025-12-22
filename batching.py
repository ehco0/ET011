import openvsp_config # type: ignore
openvsp_config.LOAD_GRAPHICS = False

import openvsp as vsp # type: ignore

import csv
import random
import os
import math

from pathlib import Path

print(vsp.__file__)

BASE_DIR = Path(__file__).parent
TEMPLATE_FILE = BASE_DIR / 'wing model' / 'b737-wing.vsp3'
WING_NAME = 'Wing'
OUT_CSV = BASE_DIR / 'batch_data.csv'
N_SAMPLES = 1067

SAVE_GEOMETRY_PER_CASE = True
GEOM_OUT_DIR = BASE_DIR / 'geom_cases'

def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

def sample_params():
    Aspect   = random.uniform(3.0, 20.0)    
    Taper    = random.uniform(0.1, 1.2)  
    Sweep    = random.uniform(-10.0, 45.0)    
    Dihedral = random.uniform(-10.0, 10.0) 
    Twist    = random.uniform(-2.0, 8.0)    
    return Aspect, Taper, Sweep, Dihedral, Twist

def sample_re():
    Re = random.uniform(1e6, 2.5e7)
    return Re


def export_live_mesh():
    vsp.ExportFile(str(BASE_DIR / 'live_wing'), vsp.SET_ALL, vsp.EXPORT_STL)

def set_wing_params(Aspect, Taper, Sweep, Dihedral, Twist):
    vsp.VSPRenew()
    vsp.ReadVSPFile(str(TEMPLATE_FILE))

    wing_id = vsp.FindGeom(WING_NAME, 0)
    if not wing_id:
        raise RuntimeError(f"Wing named '{WING_NAME}' not found.")
    
    vsp.SetParmVal(wing_id, 'Aspect', 'XSec_1', Aspect)
    vsp.SetParmVal(wing_id, 'Taper',  'XSec_1', Taper)
    vsp.SetParmVal(wing_id, 'Sweep', 'XSec_1', Sweep)
    vsp.SetParmVal(wing_id, 'Dihedral', 'XSec_1', Dihedral)
    vsp.SetParmVal(wing_id, 'Twist', 'XSec_1', Twist)

    vsp.SetParmVal(wing_id, 'TotalArea', 'WingGeom', 2094.26023)

    vsp.Update()
    return wing_id

def save_geometry_for_case(case_idx, Aspect, Taper):
    ensure_output_dir(GEOM_OUT_DIR)
    fname = f'case_{case_idx:03d}_AR{Aspect:.2f}_TR{Taper:.2f}.vsp3'
    out_path = os.path.join(GEOM_OUT_DIR, fname)
    vsp.WriteVSPFile(out_path, vsp.SET_ALL)
    print('Saved geometry to:', out_path)

def run_vspaero(alpha_start_deg, alpha_end_deg, Npts=16, mach=0.6, Re=1e6):
    
    compgeom_name = 'VSPAEROComputeGeometry'
    vsp.SetAnalysisInputDefaults(compgeom_name)
    vsp.SetIntAnalysisInput(compgeom_name, 'GeomSet', [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(compgeom_name, 'ThinGeomSet', [vsp.SET_ALL])

    print('\tExecuting VSPAEROComputeGeometry...')
    vsp.ExecAnalysis(compgeom_name)
    print('\tGeometry COMPLETE')

    analysis_name = 'VSPAEROSweep'
    vsp.SetAnalysisInputDefaults(analysis_name)

    vsp.SetIntAnalysisInput(analysis_name, 'GeomSet', [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(analysis_name, 'RefFlag', [1])
    vsp.SetIntAnalysisInput(analysis_name, 'WakeNumIter', [60])
    vsp.SetIntAnalysisInput(analysis_name, 'NumWakeNodes', [5])
    vsp.SetIntAnalysisInput(analysis_name, 'StallModel', [1])

    wing_id = vsp.FindGeom(WING_NAME, 0)
    if wing_id:
        vsp.SetStringAnalysisInput(analysis_name, 'WingID', [wing_id])

    # AoA
    vsp.SetDoubleAnalysisInput(analysis_name, 'AlphaStart', [alpha_start_deg])
    vsp.SetDoubleAnalysisInput(analysis_name, 'AlphaEnd', [alpha_end_deg])
    vsp.SetIntAnalysisInput(analysis_name, 'AlphaNpts', [Npts])

    # mach
    vsp.SetDoubleAnalysisInput(analysis_name, 'MachStart', [mach])
    vsp.SetDoubleAnalysisInput(analysis_name, 'MachEnd', [mach])
    vsp.SetIntAnalysisInput(analysis_name, 'MachNpts', [1])

    # reynolds
    vsp.SetDoubleAnalysisInput(analysis_name, 'ReCref', [Re])
    vsp.SetDoubleAnalysisInput(analysis_name, 'ReCrefEnd', [Re])
    vsp.SetIntAnalysisInput(analysis_name, 'ReCrefNpts', [1])

    print('\tExecuting VSPAEROSweep...')
    res_id = vsp.ExecAnalysis(analysis_name)
    print('\tSweep COMPLETE')

    # polar result
    polar_id = None
    for rid in vsp.GetStringResults(res_id, 'ResultsVec'):
        if vsp.GetResultsName(rid) == 'VSPAERO_Polar':
            polar_id = rid
            break

    if polar_id is None:
        print('Could not find VSPAERO_Polar in ResultsVec')
        return []

    # find names used by vspaero for polar results
    data_names = set(vsp.GetAllDataNames(polar_id))
    # print('Polar data names:', data_names) 

    # extract arrays from polar
    alpha_list = vsp.GetDoubleResults(polar_id, 'Alpha') if 'Alpha' in data_names else []
    CLtot_list = vsp.GetDoubleResults(polar_id, 'CLtot') if 'CLtot' in data_names else []
    CDi_list = vsp.GetDoubleResults(polar_id, 'CDi') if 'CDi'   in data_names else []
    CDo_list = vsp.GetDoubleResults(polar_id, 'CDo') if 'CDo'   in data_names else []
    CDtot_list = vsp.GetDoubleResults(polar_id, 'CDtot') if 'CDtot' in data_names else []
    CMytot_list = vsp.GetDoubleResults(polar_id, 'CMytot') if 'CMytot' in data_names else []
    L_D_list = vsp.GetDoubleResults(polar_id, 'L_D') if 'L_D' in data_names else []

    n = min(len(alpha_list), len(CLtot_list), len(CDi_list), len(CDo_list), len(CDtot_list), len(CMytot_list), len(L_D_list))
    results = []

    for i in range(n):
        alpha = alpha_list[i]
        CLtot = CLtot_list[i]
        CDi   = CDi_list[i]
        CDo   = CDo_list[i]
        CDtot = CDtot_list[i]
        CMytot = CMytot_list[i]
        L_D   = L_D_list[i]
        results.append((alpha, CLtot, CDi, CDo, CDtot, CMytot, L_D))

    return results

def main():
    print('Working directory:', os.getcwd())
    print('Output CSV will be:', os.path.abspath(OUT_CSV))

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Aspect','Taper','Sweep','Dihedral','Twist','Re','Alpha','CLtot','CDi', 'CDo','CDtot','CMytot','L_D'])

        for i in range(N_SAMPLES):
            Aspect, Taper, Sweep, Dihedral, Twist = sample_params()
            Re = sample_re()
                        
            print(f'\nCase {i+1}/{N_SAMPLES}: Aspect={Aspect:.3f}, Taper={Taper:.3f}')

            wing_id = set_wing_params(Aspect, Taper, Sweep, Dihedral, Twist)
            export_live_mesh()

            if SAVE_GEOMETRY_PER_CASE:
                save_geometry_for_case(i+1, Aspect, Taper)

            aero_data = []
            try:
                aero_data = run_vspaero(-6.0, 24.0, Npts=16, mach=0.6, Re=Re)
            except Exception as e:
                print('VSPAERO failed:', e)

            if not aero_data:
                writer.writerow([Aspect, Taper, Sweep, Dihedral, Twist, Re,  None, None, None, None, None, None, None])
                f.flush()
                print('No aero data for this case.')
                continue

            for (Alpha, CLtot, CDi, CDo, CDtot, CMytot, L_D) in aero_data:
                writer.writerow([Aspect, Taper, Sweep, Dihedral, Twist, Re, Alpha, CLtot, CDi, CDo, CDtot, CMytot, L_D])
                print(f'ROW: AR={Aspect:.3f}, TR={Taper:.3f}, RE={Re:.3f},'
                      f'Alpha={Alpha:.3f}, CLtot={CLtot:.4f}, '
                      f'CDi={CDi:.4f}, CDo={CDo:.4f}, CDtot={CDtot:.4f}, CMytot={CMytot:.4f}, L/D={L_D:.4f}')
            f.flush()

        print('\nDone. Data saved to:', os.path.abspath(OUT_CSV))


if __name__ == '__main__':
    main()