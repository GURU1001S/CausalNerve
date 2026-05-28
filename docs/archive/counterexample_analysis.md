# Counterexample Analysis: Failure Modes of the Structural Lyapunov Gate

While Theorem 1 and Theorem 2 prove that the CausalNerve Lyapunov Gate provides absolute protection against graph oscillation storms and guarantees convergence, these strict mathematical constraints introduce distinct pathological failure modes.

## 1. Trapped Equilibria (Local Minima)
**The Problem:** Because the gate strictly requires $V(G_{t+1}) < V(G_t) - \epsilon$ for every single edit operator $o \in \mathcal{O}$, the system is performing a discrete greedy descent. 

**Counterexample:**
Consider a ground truth graph $G^*$ with edges $A \to B$ and $B \to C$. 
Suppose our current graph $G_t$ has a direct shortcut $A \to C$ and no other edges.
To reach $G^*$, we must remove $A \to C$ and add $A \to B, B \to C$. 
However, removing $A \to C$ alone drastically increases $L_{leak}$ (causal leakage), meaning $V(G_{t_{rem}}) > V(G_t)$. The Lyapunov gate rejects the edit.
Adding $A \to B$ alone increases $H_{edit}$ (density penalty) without substantially reducing $L_{leak}$ because $A \to C$ already carries the predictive signal. $V(G_{t_{add}}) > V(G_t)$. The gate rejects it.
**Result:** The system is trapped in $G_t$. It cannot cross the high-energy barrier to reach the global minimum $G^*$.

## 2. Dead Graphs (Over-Constrained Sparsity)
**The Problem:** If the sparsity weight $w_{entropy}$ is set too high, the system treats any structural addition as prohibitively expensive.
**Counterexample:** 
If the system starts with an empty graph $G_0 = \emptyset$, and the leakage from a missing edge is $1.0$, but the sparsity penalty is $1.5$, then $V(G_{add}) = V(G_0) - 1.0 + 1.5 = V(G_0) + 0.5$.
The edit is rejected. 
**Result:** The graph remains completely disconnected indefinitely, despite massive predictive failure, because the energy parameters force a "dead graph" equilibrium.

## 3. The Alternating Stochastic Landscape (Violation of Assumption)
**The Problem:** The theorems assume $V(G)$ is evaluated deterministically. In streaming systems, $L_{leak}$ is computed over a sliding temporal window of data $X_t$. Thus, the landscape is time-varying: $V_t(G)$.
**Counterexample:**
At time $t_1$, noise in sensor $A$ causes an edge $A \to B$ to appear useful. $V_{t_1}(G \cup \{A \to B\}) < V_{t_1}(G) - \epsilon$. The edge is added.
At time $t_2$, the noise dissipates. The sparsity penalty dominates. $V_{t_2}(G \setminus \{A \to B\}) < V_{t_2}(G) - \epsilon$. The edge is removed.
**Result:** The system enters a slow oscillation cycle, bounded not by the graph algorithm, but by the frequency of the environmental noise. This requires the `UncertaintyEngine` and Aleatoric variance penalties (implemented previously) to prevent.

## 4. Comparison Summary
If CausalNerve used **Simulated Annealing (SA)**, it could probabilistically accept energy increases $V(G_{prop}) > V(G_t)$ to escape the local minima in Failure Mode 1. However, SA is notoriously slow to converge. For a live, physical control system like a jet engine, allowing the causal graph to randomly explore "worse" topologies in real-time is an unacceptable safety risk.

The Lyapunov Gate explicitly accepts **sub-optimality** (getting trapped in local minima) in exchange for absolute **stability and safety** (never diverging).
