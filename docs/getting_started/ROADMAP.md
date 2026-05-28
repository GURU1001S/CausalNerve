# CausalNerve Architecture Roadmap (2026-2027)

Our goal is to make CausalNerve the de facto standard for online causal discovery in edge industrial networks.

## Phase 1: Stabilization (Current)
- [x] Detach from heavy LLMs, transition to deterministic symbolic reasoning.
- [x] Implement rigorous, reproducible scientific benchmarking.
- [x] Create modular Plugin registry for domain adaptability.
- [x] Establish high-frequency telemetry streaming layer.

## Phase 2: Scale & Memory (Q3 2026)
- [ ] **Graph Database Backend**: Migrate `FleetEpidemiologyGraph` from in-memory dicts to Neo4j to support >10,000 concurrent engines.
- [ ] **WebSockets**: Upgrade the Observatory dashboard from HTTP polling to WebSockets for ultra-low latency rendering.
- [ ] **FCI Algorithm Integration**: Support latent confounder detection using Fast Causal Inference constraints.

## Phase 3: The Multi-Agent Edge (Q1 2027)
- [ ] **Federated Causal Learning**: Allow isolated CausalNerve instances at different manufacturing plants to share abstracted "Motifs" without sharing raw proprietary telemetry data.
- [ ] **Causal Reinforcement Learning**: Directly pipe the generated `CausalNerve` graph into an offline RL agent to learn optimal asset control strategies.
