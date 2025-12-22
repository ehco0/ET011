from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import sys
import uuid
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.run_design import run_design

app = FastAPI(title="Wing Optimiser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS = {}

def landing_html():
    return """
    <!doctype html>
    <html>
    <body style="font-family:Arial; padding:20px;">
      <h2>Wing Optimiser</h2>
      <p>Select a mode:</p>
      <a href="/glider"><button style="font-size:18px; padding:10px 16px;">Glider</button></a>
      <a href="/normal"><button style="font-size:18px; padding:10px 16px; margin-left:10px;">Normal</button></a>
    </body>
    </html>
    """

def mode_html(mode: str):
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>{mode.title()} Wing Optimiser</title>
      <style>
        body {{ font-family: Arial; padding: 16px; }}
        label {{ display:block; margin: 10px 0; }}
        input {{ width: 220px; }}
        #viewerFrame {{ width: 100%; height: 650px; border: 1px solid #ddd; border-radius: 8px; margin-top: 14px; }}
        pre {{ background:#f6f6f6; padding:10px; border-radius:8px; }}
      </style>
    </head>
    <body>
      <a href="/">← back</a>
      <h2>{mode.title()} mode</h2>

      <label>Re: <input id="re" type="number" value="10000000"></label>
      <label>Alpha start: <input id="a0" type="number" value="-10"></label>
      <label>Alpha end: <input id="a1" type="number" value="20"x    ></label>

      <button id="runBtn">Run optimisation</button>
      <div id="status" style="margin-top:10px;"></div>

      <h3>Best geometry</h3>
      <pre id="geom"></pre>

      <iframe id="viewerFrame"></iframe>

      <script>
        const MODE = "{mode}";

        function fmtGeom(obj) {{
          return Object.entries(obj).map(([k,v]) => `${{k}}: ${{Number(v).toFixed(4)}}`).join("\\n");
        }}

        document.getElementById("runBtn").onclick = async () => {{
          const status = document.getElementById("status");
          status.textContent = "Running...";
          document.getElementById("geom").textContent = "";

          const payload = {{
            Re: Number(document.getElementById("re").value),
            alpha_start: Number(document.getElementById("a0").value),
            alpha_end: Number(document.getElementById("a1").value),
            mode: MODE
          }};

          const res = await fetch("/optimise", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(payload)
          }});

          const text = await res.text();
          if (!res.ok) {{
            status.textContent = "Error: " + text;
            return;
          }}

          const data = JSON.parse(text);
          status.textContent = "Done. Opening viewer...";
          document.getElementById("geom").textContent = fmtGeom(data.best_geom);
          document.getElementById("viewerFrame").src = data.viewer_url;
        }};
      </script>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
def landing():
    return HTMLResponse(landing_html())

@app.get("/glider", response_class=HTMLResponse)
def glider_page():
    return HTMLResponse(mode_html("glider"))

@app.get("/normal", response_class=HTMLResponse)
def normal_page():
    return HTMLResponse(mode_html("normal"))

class OptimiseRequest(BaseModel):
    Re: float
    alpha_start: float
    alpha_end: float
    mode: str = "glider"

import numpy as np

@app.post("/optimise")
def optimise(req: OptimiseRequest):
    run_id = str(uuid.uuid4())[:8]
    out_dir = Path("outputs_web") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_design(
            Re=req.Re,
            Alpha_start=req.alpha_start,
            Alpha_end=req.alpha_end,
            mode=req.mode,
            out_dir=str(out_dir),
        )

        def convert_ndarray(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  
            elif isinstance(obj, dict):
                return {key: convert_ndarray(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_ndarray(item) for item in obj]
            return obj

        result_serializable = convert_ndarray(result)

        stl_path = out_dir / f"{run_id}.stl"
        RUNS[run_id] = {"stl_path": stl_path, "result": result}

        return JSONResponse({
            "run_id": run_id,
            "viewer_url": f"/viewer/{run_id}",
            "stl_url": f"/runs/{run_id}/stl",
            "best_geom": result_serializable.get("best_geom", {}),
            "alpha_sweep": result_serializable.get("env", {}).get("Alpha_sweep", []),
            "aero": result_serializable.get("best_aero", {}),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/runs/{run_id}/stl")
def get_stl(run_id: str):
    if run_id not in RUNS:
        return JSONResponse({"error": "run_id not found"}, status_code=404)

    stl_path = RUNS[run_id]["stl_path"]
    
    # Debugging log to check if the file exists
    if not stl_path.exists():
        print(f"STL file not found at {stl_path}")  # Debugging log
        return JSONResponse({"error": "STL file not found"}, status_code=404)

    return FileResponse(stl_path, media_type="model/stl", filename=stl_path.name)

@app.get("/viewer/{run_id}", response_class=HTMLResponse)
def viewer(run_id: str):
    if run_id not in RUNS:
        return HTMLResponse("run_id not found", status_code=404)

    # Correct URL for the STL file in the 3D viewer
    stl_url = f"/runs/{run_id}/stl"

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>STL Viewer {run_id}</title>
      <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #viewer {{ width: 100vw; height: 100vh; }}
        .hud {{
          position: fixed; top: 12px; left: 12px;
          background: rgba(255,255,255,0.9);
          padding: 10px 12px; border-radius: 8px;
          border: 1px solid #ddd; font-size: 14px;
          z-index: 10;
        }}
      </style>
    </head>
    <body>
      <div class="hud">
        <b>Run:</b> {run_id}<br/>
        <b>STL:</b> <a href="{stl_url}" target="_blank">download/view</a>
      </div>
      <div id="viewer"></div>

      <script type="module">
        import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
        import {{ OrbitControls }} from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
        import {{ STLLoader }} from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/STLLoader.js";

        const container = document.getElementById("viewer");
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf7f7f7);

        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.001, 10000);
        camera.position.set(1.5, 1.0, 1.8);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        scene.add(new THREE.AmbientLight(0xffffff, 0.9));
        const light = new THREE.DirectionalLight(0xffffff, 0.8);
        light.position.set(2, 3, 4);
        scene.add(light);

        scene.add(new THREE.AxesHelper(0.25));

        function fitCameraToObject(obj) {{
          const box = new THREE.Box3().setFromObject(obj);
          const size = box.getSize(new THREE.Vector3());
          const center = box.getCenter(new THREE.Vector3());

          const maxDim = Math.max(size.x, size.y, size.z);
          const fov = camera.fov * (Math.PI / 180);
          let camZ = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 2.0;

          camera.position.set(center.x + camZ, center.y + camZ * 0.6, center.z + camZ);
          camera.lookAt(center);
          controls.target.copy(center);
          controls.update();
        }}

        const loader = new STLLoader();
        loader.load("{stl_url}", (geometry) => {{
          geometry.computeVertexNormals();
          const material = new THREE.MeshStandardMaterial({{ color: 0xbfbfbf, roughness: 0.85, metalness: 0.05 }});
          const mesh = new THREE.Mesh(geometry, material);

          mesh.rotation.x = -Math.PI / 2;

          scene.add(mesh);
          fitCameraToObject(mesh);
        }}, undefined, (err) => {{
          console.error(err);
          alert("Failed to load STL. Check {stl_url} endpoint.");
        }});

        function onResize() {{
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        }}
        window.addEventListener("resize", onResize);

        function animate() {{
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        }}
        animate();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)