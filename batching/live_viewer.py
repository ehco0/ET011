import pyvista as pv 
from pyvistaqt import BackgroundPlotter 
import os
import time

FILENAME = 'live_wing.stl'

print('DEBUG: starting viewer')

plotter = BackgroundPlotter()
mesh_actor = None
last_mtime = 0

print('DEBUG: viewer window created')

while plotter.active:
    if os.path.exists(FILENAME):
        mtime = os.path.getmtime(FILENAME)

        if mtime != last_mtime:
            last_mtime = mtime
            print('DEBUG: reloading mesh')

            try:
                mesh = pv.read(FILENAME)
            except Exception as e:
                print('ERROR reading STL:', e)
                time.sleep(0.1)
                continue

            # remove old mesh
            if mesh_actor is not None:
                plotter.remove_actor(mesh_actor)

            # add new mesh
            mesh_actor = plotter.add_mesh(mesh, color='white')
            plotter.reset_camera()

    # KEEP WINDOW ACTIVE
    plotter.app.processEvents()
    time.sleep(0.1)
