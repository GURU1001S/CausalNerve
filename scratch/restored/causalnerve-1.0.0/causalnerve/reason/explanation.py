"""
causalnerve.reasoning.explanation
==============================
Deterministic natural-language explanation engine for CausalNerve.
Translates structural graphs, energies, and counterfactuals into human-readable text.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .trace import TraceResult
from .counterfactual import CounterfactualResult

class CausalNarrator:
    """
    CausalNarrator generates structured narratives for alarms, interventions,
    revisions, and health reports, supporting concise, engineering, and academic modes.
    """
    
    def explain_alarm(self, 
                      node: int, 
                      leakage: float, 
                      threshold: float,
                      mode: str = "concise", 
                      node_labels: Optional[Dict[int, str]] = None) -> str:
        name = node_labels.get(node, f"Node {node}") if node_labels else f"Node {node}"
        
        if mode == "concise":
            return f"Alarm: {name} instability detected (leakage {leakage:.4f} > threshold {threshold:.4f})."
        elif mode == "engineering":
            return (f"[ALARM] {name} (ID: {node}) has breached predictive bounds.\n"
                    f"Observed prediction leakage: {leakage:.6f}\n"
                    f"System threshold limit: {threshold:.6f}\n"
                    f"Recommendation: Trigger structural trace query.")
        else: # academic
            return (f"\\mathcal{{A}}_{{{node}}}: \\text{{Instability state detected in }} {name}.\n"
                    f"Residual error variance \\epsilon_t^2 = {leakage:.4f} exceeds safety threshold \\theta = {threshold:.4f}.\n"
                    f"Rejection probability p(V(G_{{new}}) < V(G_{{old}})) = 1.0 until calibration completes.")

    def explain_root_cause(self, 
                           trace: TraceResult, 
                           mode: str = "concise", 
                           node_labels: Optional[Dict[int, str]] = None) -> str:
        target_node = trace.active_anomalies[0] if trace.active_anomalies else 0
        target_name = node_labels.get(target_node, f"Node {target_node}") if node_labels else f"Node {target_node}"
        precursor_name = node_labels.get(trace.earliest_precursor, f"Node {trace.earliest_precursor}") if node_labels else f"Node {trace.earliest_precursor}"
        
        # Format chain
        chain_names = [node_labels.get(n, f"Node {n}") if node_labels else f"Node {n}" for n in trace.causal_chain]
        chain_str = " -> ".join(chain_names)
        
        if mode == "concise":
            return (f"Root cause of {target_name} anomaly traced to {precursor_name} "
                    f"beginning {trace.precursor_latency_cycles} cycles earlier. "
                    f"Path: {chain_str} (Confidence: {trace.confidence:.2f}).")
        elif mode == "engineering":
            contrib_lines = []
            for n, pct in trace.contribution_percentages.items():
                n_name = node_labels.get(n, f"Node {n}") if node_labels else f"Node {n}"
                contrib_lines.append(f"   - {n_name} (ID: {n}): {pct*100:.1f}%")
            contrib_str = "\n".join(contrib_lines)
            
            return (f"--- Root Cause Diagnostic ---\n"
                    f"Target Anomaly: {target_name}\n"
                    f"Earliest Precursor Source: {precursor_name}\n"
                    f"Propagation Delay: {trace.precursor_latency_cycles} time cycles\n"
                    f"Causal Pathway: {chain_str}\n"
                    f"Contribution Attribution:\n{contrib_str}\n"
                    f"Trace Confidence: {trace.confidence:.4f}")
        else: # academic
            contrib_math = ", ".join([f"w_{{{n}}}={pct:.2f}" for n, pct in trace.contribution_percentages.items()])
            return (f"\\text{{Root cause explanation for }} {target_name} \\text{{ under DAG backtracking:}}\n"
                    f"\\text{{Earliest non-stationary precursor: }} X_{{{trace.earliest_precursor}}} \\text{{ ({precursor_name})}}\n"
                    f"\\text{{Structural latency: }} \\tau = {trace.precursor_latency_cycles} \\text{{ periods}}\n"
                    f"\\text{{Causal pathway: }} \\pi = \\{{ {', '.join(map(str, trace.causal_chain))} \\}}\n"
                    f"\\text{{Attributed weights: }} \\mathbf{{w}} = \\{{ {contrib_math} \\}}\n"
                    f"\\text{{Bayesian confidence: }} P(\\pi) = {trace.confidence:.4f}")

    def explain_revision(self, 
                         event: Any, 
                         mode: str = "concise", 
                         node_labels: Optional[Dict[int, str]] = None) -> str:
        # event is expected to have attributes: cycle, proposed_edge, v_before, v_after, decision, leakage_delta
        # Let's support dict or object representation
        is_dict = isinstance(event, dict)
        cycle = event.get("cycle", 0) if is_dict else getattr(event, "cycle", 0)
        edge = event.get("proposed_edge", (0, 1)) if is_dict else getattr(event, "proposed_edge", (0, 1))
        v_before = event.get("v_before", 0.0) if is_dict else getattr(event, "v_before", 0.0)
        v_after = event.get("v_after", 0.0) if is_dict else getattr(event, "v_after", 0.0)
        decision = event.get("decision", "rejected") if is_dict else getattr(event, "decision", "rejected")
        leak_delta = event.get("leakage_delta", 0.0) if is_dict else getattr(event, "leakage_delta", 0.0)
        
        u_name = node_labels.get(edge[0], f"Node {edge[0]}") if node_labels else f"Node {edge[0]}"
        v_name = node_labels.get(edge[1], f"Node {edge[1]}") if node_labels else f"Node {edge[1]}"
        
        if mode == "concise":
            return f"Cycle {cycle}: proposed {u_name}->{v_name} {decision}. V: {v_before:.2f}->{v_after:.2f} (leakage delta: {leak_delta*100:+.1f}%)."
        elif mode == "engineering":
            return (f"--- OCGR Revision Event (Cycle {cycle}) ---\n"
                    f"Proposed Edge: {u_name} -> {v_name} ({edge[0]} -> {edge[1]})\n"
                    f"Lyapunov Energy State: V_before = {v_before:.4f}, V_after = {v_after:.4f}\n"
                    f"Energy delta: {v_after - v_before:+.4f}\n"
                    f"Leakage Reduction: {leak_delta*100:+.2f}%\n"
                    f"Decision: {decision.upper()}")
        else: # academic
            return (f"\\text{{Revision Event at }} t={cycle}:\n"
                    f"\\text{{Proposed transition: }} G \\cup \\{{ X_{{{edge[0]}}} \\to X_{{{edge[1]}}} \\}}\n"
                    f"\\Delta V(G) = V(G_{{after}}) - V(G_{{before}}) = {v_after - v_before:.4f} \\quad (\\epsilon = 0.50)\n"
                    f"\\text{{Predictive leakage variation: }} \\Delta \\mathcal{{L}}_{{leak}} = {leak_delta:+.4f}\n"
                    f"\\text{{Decision Outcome: }} \\mathbb{{I}}[V_{{after}} < V_{{before}} - \\epsilon] = {1 if decision == 'accepted' else 0} \\implies \\text{{{decision.upper()}}}")

    def explain_intervention(self, 
                             result: CounterfactualResult, 
                             intervention: Dict[int, float], 
                             mode: str = "concise", 
                             node_labels: Optional[Dict[int, str]] = None) -> str:
        target_node = list(intervention.keys())[0]
        val = intervention[target_node]
        target_name = node_labels.get(target_node, f"Node {target_node}") if node_labels else f"Node {target_node}"
        
        aff = [node_labels.get(n, f"Node {n}") if node_labels else f"Node {n}" for n in result.affected_nodes]
        unaff = [node_labels.get(n, f"Node {n}") if node_labels else f"Node {n}" for n in result.unaffected_nodes]
        
        aff_str = ", ".join(aff) if aff else "None"
        unaff_str = ", ".join(unaff) if unaff else "None"
        
        div_vol = float(np.sum(np.abs(result.divergence)))
        
        diff = result.world_1_trajectory - result.world_0_trajectory
        curve = np.mean(diff ** 2, axis=(1, 2)) if diff.ndim == 3 else np.mean(diff ** 2, axis=-1)
        score = float(np.sum(curve))
        
        if mode == "concise":
            return (f"Intervening on {target_name} at {val:.2f} affects descendants [{aff_str}]. "
                    f"Divergence volume: {div_vol:.2f}.")
        elif mode == "engineering":
            return (f"--- Counterfactual Rollout Summary ---\n"
                    f"Intervention Setup: do({target_name} = {val:.4f})\n"
                    f"Downstream Descendants (Affected): {aff_str}\n"
                    f"Non-Descendants (Protected): {unaff_str}\n"
                    f"Total Counterfactual Divergence Volume: {div_vol:.6f}\n"
                    f"Intervention Value Index: {score:.4f}")
        else: # academic
            return (f"\\text{{Counterfactual Intervention do}}(X_{{{target_node}}} = {val:.2f}):\n"
                    f"\\text{{Factual distribution: }} P(\\mathbf{{X}}_t) \\quad vs \\quad \\text{{Intervened distribution: }} P(\\mathbf{{X}}_t \\mid \\text{{do}}(X_{{{target_node}}} = {val:.2f}))\n"
                    f"\\text{{Affected descendants set }} \\mathcal{{D}} = \\{{ {', '.join(aff)} \\}}\n"
                    f"\\text{{Divergence volume }} \\int_t ||\\mathbf{{X}}^{{\\text{{factual}}}} - \\mathbf{{X}}^{{\\text{{intervened}}}}||_1 dt = {div_vol:.4f}\n"
                    f"\\text{{Information efficacy score }} \\mathcal{{S}}_{{eff}} = {score:.4f}")

    def explain_divergence(self, 
                           result: CounterfactualResult, 
                           mode: str = "concise", 
                           node_labels: Optional[Dict[int, str]] = None) -> str:
        div_vol = float(np.sum(np.abs(result.divergence)))
        if mode == "concise":
            return f"Counterfactual divergence is {div_vol:.2f}."
        elif mode == "engineering":
            return f"Factual and intervened trajectories diverged by a cumulative L1 magnitude of {div_vol:.4f}."
        else:
            return f"\\text{{L1 Divergence Metric: }} D_{{L1}}(W_0, W_1) = {div_vol:.4f}"

    def summarize_graph_health(self, 
                               health_report: Any, 
                               mode: str = "concise", 
                               node_labels: Optional[Dict[int, str]] = None) -> str:
        # Health report properties: overall_leakage, alarmed_edges, v_energy, status
        is_dict = isinstance(health_report, dict)
        leakage = health_report.get("overall_leakage", 0.0) if is_dict else getattr(health_report, "overall_leakage", 0.0)
        energy = health_report.get("v_energy", 0.0) if is_dict else getattr(health_report, "v_energy", 0.0)
        status = health_report.get("status", "healthy") if is_dict else getattr(health_report, "status", "healthy")
        
        if mode == "concise":
            return f"Graph Health: {status.upper()} (Leakage: {leakage:.4f}, Energy: {energy:.2f})."
        elif mode == "engineering":
            return (f"--- Structural Graph Health ---\n"
                    f"Overall Residual Leakage: {leakage:.6f}\n"
                    f"Lyapunov Free Energy: {energy:.4f}\n"
                    f"Operational Status: {status.upper()}")
        else: # academic
            return (f"\\text{{Graph Health Summary:}}\n"
                    f"\\text{{Current state space stability: }} \\text{{{status.upper()}}}\n"
                    f"\\mathcal{{L}}_{{leak}} = {leakage:.4f} \\quad V(G) = {energy:.4f}")


class ExplanationGenerator(CausalNarrator):
    """
    Alias / Subclass to maintain backward compatibility in CausalNerve API.
    """
    def explain_anomaly(self, 
                        trace: TraceResult, 
                        anomalous_node: int, 
                        node_labels: Optional[Dict[int, str]] = None) -> str:
        # Map explain_anomaly directly to explain_root_cause in engineering mode for details
        return self.explain_root_cause(trace, mode="engineering", node_labels=node_labels)
        
    def explain_intervention_narrative(self, 
                                       result: CounterfactualResult, 
                                       intervention: Dict[int, float], 
                                       node_labels: Optional[Dict[int, str]] = None) -> str:
        return self.explain_intervention(result, intervention, mode="engineering", node_labels=node_labels)
