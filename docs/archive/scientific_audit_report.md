# Scientific Honesty Audit: CausalNerve Framework

## 1. Claim Categorization & Identification

### The Core Issue: "Causal" vs. "Adaptive Dependency"
The framework relies heavily on predictive residuals (leakage) and SVAR modeling to alter its internal directed graph. Without strict identifiability assumptions (e.g., no unobserved confounders, faithfulness, causal Markov condition) or actual interventional data during the learning phase, the system is performing **structural adaptive dependency learning**, not true causal discovery. 

- **Claim:** "CausalNerve performs true online causal discovery."
  **Status:** **Misleading / Unsupported**. It performs online *predictive dependency* discovery. It assumes that predictive leakage equates to a broken causal mechanism, which is an untestable assumption observationally.
- **Claim:** "Native do() interventions."
  **Status:** **Partially Supported**. The framework correctly implements the *mathematics* of graph surgery (severing incoming edges) for simulation, but the resulting counterfactual trajectories are only valid if the learned graph perfectly matches the true causal graph, which cannot be guaranteed from observational data alone.
- **Claim:** "Root cause analysis via tracing."
  **Status:** **Partially Supported**. It finds the topological source of an anomaly within its *learned model*. If the model suffers from unobserved confounding, the identified "root cause" may merely be a correlated precursor.
- **Claim:** "Perfect structural recovery (SHD = 0) on synthetic benchmarks."
  **Status:** **Supported (with caveats)**. It works on synthetic SVARs where the assumptions (linear additive noise, full observability) perfectly align with the generation process. It does not guarantee SHD=0 on real-world non-linear data.

---

## 2. Definitional Separation

To maintain academic credibility, we must strictly delineate the capabilities:
1. **Adaptive Structural Dependency Learning:** The OCGR engine detects structural drift in time-series predictive dependencies. It is not discovering the laws of physics from scratch.
2. **Mechanistic Reasoning:** The system provides explicit, traceable pathways through its internal state space, rather than black-box neural activations.
3. **Interventional Simulation:** The `do()` operator correctly simulates interventions *on the model's learned structure*, not necessarily the true physical world.

---

## 3. Honest Limitations & Threats to Validity

### 3.1 Unobserved Confounders (Causal Sufficiency)
The system currently assumes all relevant variables are measured. If a latent variable $U$ causes both $A$ and $B$, the system may erroneously learn the structural dependency $A \to B$. Graph surgeries performed on this edge will yield wildly inaccurate counterfactual predictions.

### 3.2 Identifiability Limits
In purely observational, continuous-time data, Markov equivalent classes often cannot be distinguished without explicit interventions. The Lyapunov descent may settle on a graph that is predictively optimal but causally reversed ($B \to A$ instead of $A \to B$).

### 3.3 The Feedback Illusion
For highly coupled, fast-feedback physical systems (like climate or turbofans), the discrete-time SVAR approximation may fail to capture continuous cyclic dynamics, leading the Dropout Artifact Detector to misclassify temporal cycles as noise.

---

## 4. Deployment Risks

Deploying this system in safety-critical environments (e.g., automated grid control or autonomous aviation) carries severe risks:
1. **The "Trusted Simulation" Trap**: Operators may over-rely on the `what_if()` counterfactual simulations. If the underlying learned topology is causally spurious, the system might predict an intervention is safe when it is actually catastrophic.
2. **Calibration Decay**: The uncertainty bounds (ECE) are calculated on historical distributions. If the system undergoes a true phase transition into an unprecedented state space, the epistemic uncertainty bounds may degrade, leading to overconfident, incorrect graph revisions.

---

## 5. Reproducibility Checklist
- [x] Are random seeds explicitly set for all synthetic data generation?
- [x] Are the SVAR noise distributions properly documented?
- [x] Is the source code for the baseline algorithms (NOTEARS, DBN) included or linked?
- [x] Are hyperparameters for the Lyapunov gate ($\epsilon$, $w_{leak}$) justified and open for ablation?
- [x] Does the benchmark report variance/confidence intervals rather than just point estimates?
