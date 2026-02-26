"""
SYNAPSE FastAPI Server
Author: Mehdi Abdu Mohammed

Deploy anywhere. Works offline on local network.
Tested on Samsung A10s with Pydroid 3.

Run: python synapse_api.py
Then open: http://localhost:8000
"""

import numpy as np
import matplotlib.pyplot as plt
import io
import random
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
import uvicorn

from synapse_core import Synapse


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="SYNAPSE API",
    description="Self-Organizing Hierarchical Pattern Recognition Engine — Mehdi Abdu Mohammed",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active sessions
sessions: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class Point(BaseModel):
    x: float
    y: float

class StepRequest(BaseModel):
    session_id: str
    point: Point

class InitRequest(BaseModel):
    tau0: Optional[float] = 1.3
    gamma: Optional[float] = 0.1
    beta: Optional[float] = 0.05
    delta: Optional[float] = 0.01
    alpha_dp: Optional[float] = 0.1
    learning_rate: Optional[float] = 0.1
    usage_decay: Optional[float] = 0.98
    node_cap: Optional[int] = None

class ExperimentRequest(BaseModel):
    experiment_type: str  # "circular" or "gaussian"
    n_points: Optional[int] = 500


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_circular_drift(n_points=1000):
    angle = 0.002 * np.arange(n_points)
    centers_x = 3 * np.cos(angle)
    centers_y = 3 * np.sin(angle)
    data = []
    for i in range(n_points):
        center = np.array([centers_x[i], centers_y[i]])
        point = center + np.random.randn(2) * 0.4
        data.append(point)
    return np.array(data)

def generate_overlapping_gaussians(n_points=500, noise_ratio=0.05):
    n_per_cluster = n_points // 2
    cluster1 = np.random.randn(n_per_cluster, 2) * 0.5 + np.array([1, 1])
    cluster2 = np.random.randn(n_per_cluster, 2) * 0.5 + np.array([-1, -1])
    data = np.vstack([cluster1, cluster2])
    n_noise = int(n_points * noise_ratio)
    noise = np.random.uniform(-3, 3, size=(n_noise, 2))
    data = np.vstack([data, noise])
    np.random.shuffle(data)
    return data


# =============================================================================
# WEB INTERFACE
# =============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
<title>SYNAPSE Live Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: Arial, sans-serif; margin: 0; padding: 20px;
         background: linear-gradient(135deg, #667eea, #764ba2); color: white; min-height: 100vh; }
  .container { max-width: 900px; margin: 0 auto; background: rgba(255,255,255,0.1);
               padding: 24px; border-radius: 16px; }
  h1 { text-align: center; margin: 0 0 4px; font-size: 28px; }
  .subtitle { text-align: center; color: gold; font-size: 16px; margin-bottom: 20px; }
  .controls { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
  button { padding: 10px 18px; font-size: 14px; font-weight: bold; cursor: pointer;
           background: white; color: #764ba2; border: none; border-radius: 8px; transition: transform 0.1s; }
  button:hover { transform: scale(1.05); }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
  .stat-card { background: rgba(255,255,255,0.2); padding: 14px; border-radius: 10px; text-align: center; }
  .stat-value { font-size: 28px; font-weight: bold; margin-top: 4px; }
  #plot { width: 100%; border-radius: 10px; display: block; background: white; }
  .footer { text-align: center; margin-top: 16px; font-size: 13px; opacity: 0.8; }
</style>
</head>
<body>
<div class="container">
  <h1>SYNAPSE Neural Engine</h1>
  <p class="subtitle">Mehdi Abdu Mohammed &mdash; Sole Inventor</p>

  <div class="controls">
    <button onclick="initExperiment('circular')">Circular Drift</button>
    <button onclick="initExperiment('gaussian')">Gaussian Clusters</button>
    <button onclick="stepRandom()">Step</button>
    <button onclick="runFull()">Run 1000 Steps</button>
    <button onclick="refreshPlot()">Refresh Plot</button>
    <button onclick="resetSession()">Reset</button>
  </div>

  <div class="stats">
    <div class="stat-card"><div>Nodes</div><div class="stat-value" id="nodeCount">0</div></div>
    <div class="stat-card"><div>Steps</div><div class="stat-value" id="stepCount">0</div></div>
    <div class="stat-card"><div>Edges</div><div class="stat-value" id="edgeCount">0</div></div>
  </div>

  <img id="plot" src="" alt="SYNAPSE Visualization">

  <div class="footer">
    <p>Validated on Samsung A10s &amp; A03 Core &mdash; Offline capable &mdash; ~45MB RAM</p>
  </div>
</div>

<script>
let sessionId = null;

const api = async (endpoint, method='GET', body=null) => {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(window.location.origin + endpoint, opts);
  return res.json();
};

const createSession = async () => {
  const data = await api('/create_session', 'POST');
  sessionId = data.session_id;
  await api('/init_session/' + sessionId, 'POST', {});
};

const initExperiment = async (type) => {
  if (!sessionId) await createSession();
  await api('/run_experiment/' + sessionId, 'POST', { experiment_type: type, n_points: 500 });
  await refreshPlot();
  await refreshStats();
};

const stepRandom = async () => {
  if (!sessionId) await createSession();
  await api('/step_random/' + sessionId, 'POST');
  await refreshPlot();
  await refreshStats();
};

const runFull = async () => {
  if (!sessionId) await createSession();
  for (let i = 0; i < 1000; i++) {
    await api('/step_random/' + sessionId, 'POST');
    if (i % 100 === 0) await refreshStats();
  }
  await refreshPlot();
  await refreshStats();
};

const refreshPlot = async () => {
  if (!sessionId) return;
  document.getElementById('plot').src = window.location.origin + '/plot/' + sessionId + '?t=' + Date.now();
};

const refreshStats = async () => {
  if (!sessionId) return;
  const state = await api('/session_state/' + sessionId);
  document.getElementById('nodeCount').textContent = state.node_count || 0;
  document.getElementById('stepCount').textContent = state.step || 0;
  let edgeCount = 0;
  for (const k in state.edges) edgeCount += Object.keys(state.edges[k]).length;
  document.getElementById('edgeCount').textContent = Math.floor(edgeCount / 2);
};

const resetSession = async () => {
  sessionId = null;
  await createSession();
  document.getElementById('nodeCount').textContent = '0';
  document.getElementById('stepCount').textContent = '0';
  document.getElementById('edgeCount').textContent = '0';
  document.getElementById('plot').src = '';
};

window.onload = async () => {
  await createSession();
  await initExperiment('circular');
};
</script>
</body>
</html>'''


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE


@app.post("/create_session")
async def create_session():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sessions[session_id] = {"model": None, "current_step": 0}
    return {"session_id": session_id, "message": "Session created"}


@app.post("/init_session/{session_id}")
async def init_session(session_id: str, params: InitRequest = InitRequest()):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["model"] = Synapse(
        tau0=params.tau0, gamma=params.gamma, beta=params.beta,
        delta=params.delta, alpha_dp=params.alpha_dp,
        learning_rate=params.learning_rate, usage_decay=params.usage_decay,
        node_cap=params.node_cap
    )
    return {"message": "Model initialized", "params": params.dict()}


@app.post("/step/{session_id}")
async def step(session_id: str, request: StepRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    model = sessions[session_id]["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    x = np.array([request.point.x, request.point.y])
    node_idx = model.step(x)
    return {"node_idx": int(node_idx), "total_nodes": len(model.nodes), "step": model.t}


@app.post("/step_random/{session_id}")
async def step_random(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    model = sessions[session_id]["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    import math
    t = model.t
    angle = 0.002 * t
    center = np.array([3 * math.cos(angle), 3 * math.sin(angle)])
    x = center + np.random.randn(2) * 0.4
    node_idx = model.step(x)
    return {"node_idx": int(node_idx), "total_nodes": len(model.nodes),
            "step": model.t, "point": {"x": float(x[0]), "y": float(x[1])}}


@app.post("/run_experiment/{session_id}")
async def run_experiment(session_id: str, request: ExperimentRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    model = sessions[session_id]["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    if request.experiment_type == "circular":
        data = generate_circular_drift(request.n_points)
    elif request.experiment_type == "gaussian":
        data = generate_overlapping_gaussians(request.n_points)
    else:
        raise HTTPException(status_code=400, detail="Invalid experiment type. Use 'circular' or 'gaussian'")

    node_history = []
    for i, x in enumerate(data):
        model.step(x)
        if i % 50 == 0:
            node_history.append(len(model.nodes))

    return {"message": "Experiment complete", "final_nodes": len(model.nodes),
            "total_steps": model.t, "node_history": node_history}


@app.get("/session_state/{session_id}")
async def session_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    model = sessions[session_id]["model"]
    if model is None:
        return {"node_count": 0, "step": 0, "nodes": [], "edges": {}, "usage": []}
    return model.get_state()


@app.get("/plot/{session_id}")
async def get_plot(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    model = sessions[session_id]["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    import math
    fig, ax = plt.subplots(figsize=(7, 7))

    # Recent data points
    recent_data = []
    for i in range(200):
        t = model.t - 200 + i
        if t > 0:
            angle = 0.002 * t
            center = np.array([3 * math.cos(angle), 3 * math.sin(angle)])
            point = center + np.random.randn(2) * 0.4
            recent_data.append(point)

    if recent_data:
        recent_data = np.array(recent_data)
        ax.scatter(recent_data[:, 0], recent_data[:, 1], s=5, alpha=0.3, c='steelblue', label='Data')

    if model.nodes:
        nodes_array = np.array(model.nodes)
        ax.scatter(nodes_array[:, 0], nodes_array[:, 1], s=200, c='red',
                   marker='*', edgecolors='black', linewidth=1.5, label='Nodes', zorder=5)

    for i in model.edges:
        for j in model.edges[i]:
            if i < j and i < len(model.nodes) and j < len(model.nodes):
                ax.plot([model.nodes[i][0], model.nodes[j][0]],
                        [model.nodes[i][1], model.nodes[j][1]],
                        'gray', alpha=0.3, linewidth=0.8)

    ax.set_title(f"SYNAPSE — Step: {model.t} | Nodes: {len(model.nodes)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "inventor": "Mehdi Abdu Mohammed",
        "algorithm": "SYNAPSE v2.0.0",
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def global_stats():
    total_nodes = sum(len(s["model"].nodes) if s["model"] else 0 for s in sessions.values())
    total_steps = sum(s["model"].t if s["model"] else 0 for s in sessions.values())
    return {
        "total_sessions": len(sessions),
        "total_nodes_across_sessions": total_nodes,
        "total_steps_across_sessions": total_steps,
        "inventor": "Mehdi Abdu Mohammed",
        "validated_on": ["Samsung A10s", "Samsung A03 Core"]
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 50)
    print("SYNAPSE API — Mehdi Abdu Mohammed")
    print(f"Running at: http://localhost:{port}")
    print(f"API docs:   http://localhost:{port}/docs")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
