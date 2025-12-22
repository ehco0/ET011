import openvsp as vsp  # type: ignore
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE = os.path.join(SCRIPT_DIR, "b737-wing.vsp3")   # adjust name
WING_NAME = "Wing"  # whatever you named it in GUI

vsp.VSPRenew()
err = vsp.ReadVSPFile(TEMPLATE_FILE)
print("ReadVSPFile err code:", err)

wing_id = vsp.FindGeom(WING_NAME, 0)
print("Wing ID:", wing_id)

parm_ids = vsp.GetGeomParmIDs(wing_id)

for pid in parm_ids:
    name  = vsp.GetParmName(pid)
    group = vsp.GetParmGroupName(pid)
    val   = vsp.GetParmVal(pid)
    print(f"{group:15s}  {name:20s}  {val}")

analysis_name = "VSPAEROSweep"

in_names =  vsp.GetAnalysisInputNames( analysis_name )

print("Analysis Inputs: ")

for i in range(int( len(in_names) )):

    print( ( "\t" + in_names[i] + "\n" ) )