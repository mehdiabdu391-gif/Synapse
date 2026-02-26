# SYNAPSE — Self-Organizing Hierarchical Pattern Recognition Engine

**Built by Mehdi Abdu Mohammed — Solo, Ethiopia, on a $80 Android phone.**

---

## What is SYNAPSE?

SYNAPSE is a lightweight, offline, adaptive pattern recognition engine that runs on the cheapest Android hardware available. It learns continuously from streaming data, organizes itself, prunes what it doesn't need, and works without any internet connection, cloud service, or GPU.

It was built alone, tested on a Samsung A03 Core and A10s using Pydroid 3, and validated under extreme stress conditions. If it runs there, it runs anywhere.

---

## Why does this exist?

Most machine learning systems require:
- Cloud connectivity
- High-end hardware
- Pre-labelled datasets
- Retraining when data changes

SYNAPSE requires none of these. It was built with one constraint in mind: **work on the cheapest phone possible, offline, in real time.**

That constraint makes it useful everywhere else too.

---

## What makes it different?

SYNAPSE unifies six established mathematical frameworks into a single deployable engine — something that hasn't been packaged this way before:

- **Robbins-Monro Stochastic Approximation** — provably convergent online learning
- **Manifold Regularization** — geometric alignment via graph Laplacian
- **Adaptive Resonance Theory** — novelty detection and stability
- **Dirichlet Process Mixture Models** — Bayesian nonparametric node creation
- **Minimum Description Length (MDL)** — optimal pruning of redundant structure
- **Birth-Death Edge Dynamics** — sparse, self-organizing connectivity

No single existing system (SOM, GNG, ART, online DPMM) combines all of these into one lightweight, offline-deployable engine.

---

## Hardware it runs on

| Device | RAM | Processor | Performance |
|--------|-----|-----------|-------------|
| Samsung A03 Core | 2GB | MediaTek | ~360 steps/sec |
| Samsung A10s | 2GB | Exynos 7884 | ~850 steps/sec |

Memory usage: ~45MB for 1000 data points  
CPU peak: ~32%  
Battery impact: minimal (tested at 19% battery remaining, still stable)

---

## Quick Start

### Install dependencies

```bash
pip install numpy matplotlib
```

### Basic usage

```python
from synapse import Synapse

model = Synapse()

# Feed it data points one at a time
for data_point in your_stream:
    node_idx = model.step(data_point)

# Get current state
state = model.get_state()
print(f"Nodes: {state['node_count']}, Step: {state['step']}")
```

### Key parameters

```python
model = Synapse(
    dim=2,           # Input dimensionality
    tau0=1.3,        # Base resonance threshold — higher = harder to create nodes
    gamma=0.1,       # Manifold regularization strength
    beta=0.05,       # Edge birth rate
    delta=0.01,      # Edge decay rate
    alpha_dp=0.1,    # Dirichlet concentration — lower = less random node creation
    learning_rate=0.1
)
```

For noisy/chaotic data, use tighter parameters:

```python
model = Synapse(tau0=2.2, alpha_dp=0.03, usage_decay=0.92)
```

---

## REST API (FastAPI)

SYNAPSE ships with a full REST API for integration into any system.

### Run the server

```bash
pip install fastapi uvicorn pydantic
python main.py
# Running at http://localhost:8000
```

### Available endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface with live visualization |
| POST | `/create_session` | Create new session |
| POST | `/init_session/{id}` | Initialize model with parameters |
| POST | `/step/{id}` | Process one data point |
| POST | `/run_experiment/{id}` | Run full experiment |
| GET | `/session_state/{id}` | Get current state |
| GET | `/plot/{id}` | Get PNG visualization |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation |

### Example curl usage

```bash
# Create session
curl -X POST http://localhost:8000/create_session

# Initialize
curl -X POST http://localhost:8000/init_session/SESSION_ID \
  -H "Content-Type: application/json" \
  -d '{"tau0":1.3,"gamma":0.1,"beta":0.05,"delta":0.01,"alpha_dp":0.1}'

# Feed a data point
curl -X POST http://localhost:8000/step/SESSION_ID \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID","point":{"x":1.2,"y":3.4}}'
```

---

## What it does under the hood

SYNAPSE processes streaming data points one at a time:

1. **Novelty check** — is this point close enough to an existing node?
2. **Node creation** — if not, create one (Dirichlet process decides probability)
3. **Winner selection** — find the closest resonating node
4. **Update** — move the winner toward the new point (Robbins-Monro)
5. **Edge dynamics** — strengthen connections between co-activated nodes, decay unused ones
6. **Pruning** — every N steps, remove nodes that aren't earning their keep (MDL)

The result: a self-organizing graph that tracks the shape of your data in real time, forgets what's no longer relevant, and never grows out of control.

---

## Stress test results

Tested under **EXTREME** mode — rotating through pure chaos, fast cluster switching, exploding drift, and burst noise simultaneously:

- Nodes stabilized between 10-25 under controlled parameters
- 57 nodes created under maximum chaos, pruned back to 11 within 33 steps
- Edge count self-regulated from 338 down to 36 automatically
- FPS held at 345-360 throughout
- Memory constant at ~45MB
- No crashes, no memory leaks

---

## Domains where this fits

SYNAPSE is domain-agnostic. Any streaming data that needs real-time adaptive pattern recognition is a candidate:

**Medical** — ECG/EEG anomaly detection, wearable health monitoring, seizure prediction  
**Industrial** — Predictive maintenance, vibration anomaly, quality control  
**IoT** — Sensor networks, smart agriculture, structural health monitoring  
**Defense** — Perimeter intrusion, acoustic threat ID, signal pattern analysis  
**Finance** — Real-time fraud detection, transaction anomaly  
**Environmental** — Air quality, seismic monitoring, wildlife tracking  
**Robotics** — Joint anomaly detection, adaptive motion learning  
**Telecommunications** — Network traffic anomaly, spectrum monitoring  

The engine doesn't know or care what your data means. It just finds the patterns.

---

## Versions

| Version | Description |
|---------|-------------|
| v1.0 | Core algorithm, validated on circular drift and overlapping Gaussians |
| v2.0-STRESS | Stress test mode, no node cap, maximum chaos testing |
| v2.0-INTENSE | Controlled chaos, node cap of 25, tuned for stability under noise |
| API v1.0 | FastAPI REST service with session management and live visualization |

---

## Mathematical guarantees

- **Convergence**: Learning rate η_t = η₀/(1+λt) satisfies Robbins-Monro conditions → almost sure convergence
- **Sparsity**: Expected node degree bounded by β·p/(β·p+δ)
- **Optimality**: MDL pruning maximizes posterior P(model|data) (Bayesian equivalence)
- **Distance preservation**: Johnson-Lindenstrauss projection with ε ≈ 0.316 for k=10

---

## Author

**Mehdi Abdu Mohammed**  
Sole inventor and owner  
Ethiopia  
February 2026

Built with curiosity, an Android phone, and Pydroid 3.  
No ML background. No team. No funding. Just time and questions.

---

## License

Copyright © 2026 Mehdi Abdu Mohammed. All rights reserved.

For licensing inquiries, collaboration, or integration partnerships, open an issue or reach out directly.

---

*"As long as it works on the cheapest phone, I don't need to benchmark it. Let the users choose what they want."*
