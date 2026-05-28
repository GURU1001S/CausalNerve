# CausalNerve Open-Source Launch Assets

This file contains copy-pasteable assets for sharing CausalNerve across technical platforms (Twitter/X, LinkedIn, Reddit, YouTube).

---

## 1. Twitter/X Teaser Thread (5 Posts)

### Tweet 1/5 (The Hook)
🧠 Announcing CausalNerve: A next-generation Causal Intelligence platform for non-stationary dynamical systems.

Tired of static Bayesian networks and batch causal discovery failing when sensor streams drift? We built an active structural dependency learning library.

Check out the code: https://github.com/guru-s/CausalNerve

🧵 (1/5)

### Tweet 2/5 (How it Works)
Most models assume stationary causal graphs. But in the real world:
- Turbofans degrade ⚙️
- Brain functional connectivity shifts 🧠
- Financial markets pivot 📈

CausalNerve dynamically rewrites its graph under a structural Lyapunov energy formulation. (2/5)

### Tweet 3/5 (The Tech Stack)
CausalNerve combines:
1. **Dynamic GNNs**: Localized Jacobian tracking.
2. **Structural Alarm System**: Expected Calibration Error (ECE) monitoring.
3. **Graph Surgery**: Active edge addition/severing under a strict Lyapunov convergence constraint.
4. **Dual-World Counterfactuals**: do(intervention) vs factual simulation. (3/5)

### Tweet 4/5 (Scientifically Honest Benchmarks)
No artificial perfection. Our benchmark suite evaluates CausalNerve over 20 random seeds against NOTEARS & DBNs. We model real noise, partial detections, and cataloged failures.
Read our limitations audit: https://github.com/guru-s/CausalNerve/blob/main/FAILURES.md (4/5)

### Tweet 5/5 (Quickstart)
Get started in 3 lines:
```bash
pip install causalnerve
```
```python
from causalnerve.api import CausalNerve
nerve = CausalNerve(nodes=6)
nerve.step(sensor_reading)
```
Read the docs and give us a ⭐️: https://github.com/guru-s/CausalNerve (5/5)

---

## 2. LinkedIn Technical Post

**Headline: Moving Beyond Static Causal Discovery: Introducing CausalNerve 🧠**

Most industrial AI systems rely on static graphs. When physical processes drift or degrade—like a turbofan engine turbine wearing down—traditional causal models collapse.

Today, we are open-sourcing **CausalNerve**, a python library designed for **Active Structural Dependency Learning** in non-stationary streaming systems.

Instead of retraining large models from scratch periodically, CausalNerve implements a continuous **Graph Surgery** pipeline:
1. **Detects anomalies** using structural predictive leakage and Expected Calibration Error (ECE).
2. **Proposes graph revisions** (edge addition or removal).
3. **Filters proposals** using a Structural Lyapunov Gate to guarantee asymptotic convergence.
4. **Traces root causes** and simulates **dual-world counterfactuals** in real-time.

We are committed to scientific honesty: our benchmarks are run across 20 random seeds and document known corner cases (calibration collapse, regime ambiguity, and scaling bottlenecks).

GitHub Repository: https://github.com/guru-s/CausalNerve
Documentation & Code: https://github.com/guru-s/CausalNerve

#MachineLearning #CausalInference #ArtificialIntelligence #DataScience #OpenSource

---

## 3. Reddit Launch Post (r/MachineLearning)

**Title: [Project] CausalNerve: An Open-Source Library for Active Causal Graph Surgery in Non-Stationary Streaming Systems**

Hi r/MachineLearning,

We are excited to share **CausalNerve**, a library we’ve developed for streaming causal dependency learning under structural drift.

### The Problem
Traditional causal discovery methods (like NOTEARS, FCI, or Dynamic Bayesian Networks) assume the underlying causal graph $G$ is static. In streaming real-world applications (turbofan telemetry, brain functional connectivity, financial regimes), relationships change dynamically. Retraining models from scratch is computationally expensive and introduces severe lag.

### Our Approach
CausalNerve maintains a dynamic graph representation. It implements:
- **Expected Calibration Error (ECE) Monitoring** for online anomaly alerts.
- **Lyapunov Energy Constraint**: Ensures that any proposed graph surgery (adding or severing edges) decreases the overall system energy state, bounding model oscillations.
- **Natural-Language Reasoning Engine**: Backtracks through the reversed graph to trace root causes, returning attribution percentages and delayed timelines.

### Try It Now
Install via pip:
```bash
pip install causalnerve
```

Run a simple turbofan degradation demo:
```python
from causalnerve.api import CausalNerve

nerve = CausalNerve(nodes=6)
nerve.node_labels = {0: "Fuel", 1: "N1 Speed", 2: "N2 Speed", 3: "EGT Temp", 4: "Vibration", 5: "Oil"}

# Stream step
nerve.step(sensor_state)

# Trace root cause of temperature spike
why_res = nerve.why("EGT Temp")
print(why_res.explanation)
```

We’d love to get your feedback and issues on GitHub!
Repository: https://github.com/guru-s/CausalNerve

---

## 4. 30-Second Teaser Script & YouTube Demo Outline

### 30-Second Script (Voiceover)
*(Visual: High-tech rotating neural network transition)*
"Real-world data is non-stationary. Systems drift, engines degrade, markets shift."
*(Visual: Causal graph showing an edge flash red and disconnect)*
"When the graph changes, static AI models fail. Retraining introduces lag."
*(Visual: Zoom in on CausalNerve live web demo adapting)*
"Meet CausalNerve. An open-source platform for active causal graph surgery. It detects structural anomalies, validates updates with Lyapunov stability, and traces root causes in natural language."
*(Visual: Code screen running pip install causalnerve)*
"Get started in three lines of code. Visit CausalNerve on GitHub today."

### YouTube Demo Outline (3 Minutes)
1. **0:00 - 0:30 (Hook & Problem)**: Introduce the concept of non-stationarity. Show why static GNNs and DBNs fail to adapt to structural changes.
2. **0:30 - 1:15 (CausalNerve Core Architecture)**: Explain the Lyapunov Gate and active graph surgery (adding and severing edges).
3. **1:15 - 2:15 (Live Demo Walkthrough)**: Run `examples/turbofan_demo.py` and show the Gradio dashboard interface (`app.py`). Inject an anomaly and show the causal backtracking explanations.
4. **2:15 - 3:00 (Benchmarks & Conclusion)**: Walk through the 20-seed benchmark table, highlight key results and limitations (`FAILURES.md`), and invite contributors to star the project.
