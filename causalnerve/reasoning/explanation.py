"""
causalnerve.reasoning.explanation
==============================
Deterministic natural-language explanation engine for CausalNerve.
Translates structural graphs, energies, and counterfactuals into human-readable text.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from causalnerve.interventions.trace import TraceResult
from causalnerve.interventions.counterfactual import CounterfactualResult

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

class RuntimeNarrator:
    """
    Generates human-readable explanations of runtime states.
    All output is deterministic given the same inputs.
    Used by the dashboard for real-time annotation.
    """
    
    def narrate_metric(self,
                        metric_name: str,
                        value: float,
                        threshold: Optional[float],
                        trend: str,
                        context: Optional[Dict] = None
                        ) -> str:
        
        if metric_name.lower() == 'leakage':
            if trend == 'rising' and threshold and value > threshold:
                return "Causal leakage is rising and has exceeded the alarm threshold — structural revision may be needed."
            elif trend == 'rising' and threshold and value <= threshold:
                return f"Leakage is rising but remains below the alarm threshold (current: {value:.3f}, threshold: {threshold:.3f})."
            elif trend == 'stable' and threshold and value <= threshold:
                return "Leakage is stable and within normal bounds."
            elif trend == 'falling':
                return "Leakage is decreasing — structural adaptation is working."
            else:
                return f"Leakage is {value:.3f}."
                
        elif metric_name.lower() == 'lyapunov':
            if trend == 'falling':
                return "Graph energy is converging toward structural equilibrium."
            elif trend == 'stable':
                return "Graph energy is stable — no new structural pressure."
            elif trend == 'rising':
                return "WARNING: Graph energy is rising. Lyapunov constraint may be under stress."
            else:
                return f"Graph energy is {value:.2f}."
                
        elif metric_name.lower() == 'ece':
            if value < 0.15:
                return "Confidence estimates are well-calibrated."
            elif value <= 0.25:
                return "Calibration is marginal — treat confidence scores with moderate caution."
            else:
                return "WARNING: Calibration has degraded. Confidence scores are unreliable. Review recent edits."
                
        elif metric_name.lower() == 'fsr':
            if value < 0.20:
                return "False surgery rate is within acceptable bounds."
            elif value <= 0.30:
                return "False surgery rate is elevated. Sensor corruption may be present."
            else:
                return "WARNING: False surgery rate is high. Consider increasing confidence threshold."
                
        elif metric_name.lower() == 'throughput':
            if value > 100:
                return f"Stream processing is optimal at {value:.0f} obs/sec."
            else:
                return f"Stream processing is degraded at {value:.0f} obs/sec."
                
        elif metric_name.lower() == 'revision_rate':
            if value < 5:
                return "Graph revision rate is low and stable."
            else:
                return f"High structural volatility: {value:.1f} edits/100 cycles."
                
        elif metric_name.lower() == 'active_nodes':
            return f"The active causal network contains {int(value)} nodes."
            
        elif metric_name.lower() == 'drift_score':
            if trend == 'rising':
                return "Structural drift is accelerating."
            elif trend == 'falling':
                return "Structural drift is subsiding."
            else:
                return "Structural drift is stable."
                
        elif metric_name.lower() == 'uncertainty':
            if value < 0.2:
                return "Epistemic uncertainty is contained."
            else:
                return "High epistemic uncertainty observed across predictions."
                
        elif metric_name.lower() == 'anomaly_rate':
            if value == 0:
                return "No structural anomalies detected recently."
            else:
                return f"Anomaly rate stands at {value:.1f} per 100 cycles."
                
        elif metric_name.lower() == 'rollback_count':
            if value == 0:
                return "No structural reversals recorded."
            else:
                return f"{int(value)} edits have been rolled back recently."
                
        elif metric_name.lower() == 'memory':
            return f"Memory usage is stable at {value:.1f} MB."
            
        elif metric_name.lower() == 'runtime':
            return f"Average step latency is {value:.1f} ms."
            
        else:
            return f"{metric_name} is currently {value:.2f}."
    
    def narrate_intervention(self,
                              result: Any,
                              node_labels: Dict[int, str]
                              ) -> str:
        # Accommodate CounterfactualResult or dict
        is_dict = isinstance(result, dict)
        
        target = result.get('target', 0) if is_dict else getattr(result, 'target', 0)
        value = result.get('value', 0.0) if is_dict else getattr(result, 'value', 0.0)
        horizon = result.get('horizon', 50) if is_dict else getattr(result, 'horizon', 50)
        aff = result.get('affected_nodes', []) if is_dict else getattr(result, 'affected_nodes', [])
        unaff = result.get('unaffected_nodes', []) if is_dict else getattr(result, 'unaffected_nodes', [])
        div = result.get('divergence', []) if is_dict else getattr(result, 'divergence', [])
        
        source_label = node_labels.get(target, f"Node {target}")
        affected_labels = ", ".join(node_labels.get(n, f"Node {n}") for n in aff)
        unaffected_labels = [node_labels.get(n, f"Node {n}") for n in unaff]
        
        # Calculate peak divergence
        peak_div = 0.0
        peak_cycle = 0
        if div is not None and len(div) > 0:
            if isinstance(div, list):
                peak_div = max(div)
                peak_cycle = div.index(peak_div)
            elif hasattr(div, 'max'):
                peak_div = float(np.max(np.abs(div)))
                peak_cycle = int(np.argmax(np.sum(np.abs(div), axis=1)) if div.ndim > 1 else np.argmax(np.abs(div)))

        s1 = f"Intervening on {source_label} at value {value:.2f} over a {horizon}-cycle horizon."
        
        s2 = ""
        if aff:
            s2 = f"This affects {len(aff)} downstream nodes: {affected_labels}. Peak divergence of {peak_div:.3f} occurs at cycle +{peak_cycle}."
        else:
            s2 = "This affects 0 downstream nodes."
            
        s3 = ""
        if unaff:
            unaff_preview = ", ".join(unaffected_labels[:3])
            ellipses = "..." if len(unaffected_labels) > 3 else ""
            s3 = f"{len(unaff)} nodes are confirmed non-descendants and remain unchanged: {unaff_preview}{ellipses}."
        
        s4 = ""
        if peak_div > 0.5:
            s4 = "This is a high-impact intervention."
        elif peak_div > 0.2:
            s4 = "This is a moderate-impact intervention."
        else:
            s4 = "This intervention has limited structural impact."
            
        return f"{s1} {s2} {s3} {s4}".strip()
    
    def narrate_anomaly(self,
                         trace_result: Any,
                         node_labels: Dict[int, str]
                         ) -> str:
        is_dict = isinstance(trace_result, dict)
        
        anom_node = trace_result.get('anomalous_node', 0) if is_dict else getattr(trace_result, 'active_anomalies', [0])[0]
        dev = trace_result.get('deviation', 0.0) if is_dict else getattr(trace_result, 'deviation', 0.0)
        root = trace_result.get('earliest_precursor', 0) if is_dict else getattr(trace_result, 'earliest_precursor', 0)
        contribs = trace_result.get('contribution_percentages', {root: 1.0}) if is_dict else getattr(trace_result, 'contribution_percentages', {root: 1.0})
        chain = trace_result.get('causal_chain', []) if is_dict else getattr(trace_result, 'causal_chain', [])
        conf = trace_result.get('confidence', 0.0) if is_dict else getattr(trace_result, 'confidence', 0.0)
        
        anomalous_label = node_labels.get(anom_node, f"Node {anom_node}")
        root_cause_label = node_labels.get(root, f"Node {root}")
        contribution = contribs.get(root, 0.0)
        pathway_labels = [node_labels.get(n, f"Node {n}") for n in chain]
        path_str = " -> ".join(pathway_labels) if pathway_labels else root_cause_label
        
        s1 = f"{anomalous_label} is behaving anomalously (state deviation: {dev:.2f})."
        s2 = f"Root cause analysis identifies {root_cause_label} as the likely origin ({contribution:.0%} of observed anomaly) via pathway: {path_str}."
        
        if conf > 0.7:
            s3 = "Root cause identification is high confidence."
        elif conf > 0.4:
            s3 = "Root cause identification is moderate confidence. Consider validating with a direct intervention."
        else:
            s3 = "Root cause is ambiguous. Multiple paths show similar contribution scores."
            
        return f"{s1} {s2} {s3}"
    
    def narrate_revision(self,
                          event: Any,
                          node_labels: Dict[int, str]
                          ) -> str:
        is_dict = isinstance(event, dict)
        
        edge = event.get('proposed_edge', (0, 1)) if is_dict else getattr(event, 'proposed_edge', (0, 1))
        conf = event.get('confidence', 0.0) if is_dict else getattr(event, 'confidence', 0.0)
        decision = event.get('decision', 'hold') if is_dict else getattr(event, 'decision', 'hold')
        v_delta = event.get('v_delta', 0.0) if is_dict else getattr(event, 'v_delta', 0.0)
        rationale = event.get('rationale', 'leakage reduction') if is_dict else getattr(event, 'rationale', 'leakage reduction')
        reason = event.get('rejection_reason', 'insufficient confidence') if is_dict else getattr(event, 'rejection_reason', 'insufficient confidence')
        threshold = event.get('threshold', 0.5) if is_dict else getattr(event, 'threshold', 0.5)
        
        src_label = node_labels.get(edge[0], f"Node {edge[0]}")
        dst_label = node_labels.get(edge[1], f"Node {edge[1]}")
        
        if decision == 'accepted':
            return f"Structural revision accepted: added {src_label} -> {dst_label} (confidence {conf:.2f}, V decreased by {v_delta:.2f}, rationale: {rationale})."
        elif decision == 'rejected':
            return f"Structural revision rejected: proposed {src_label} -> {dst_label} was rejected because {reason}."
        else:
            return f"Structural revision on hold: {src_label} -> {dst_label} awaiting confirmation (current confidence {conf:.2f}, need {threshold:.2f})."
    
    def narrate_safety_status(self,
                               safeguards: Dict[str, bool],
                               ece: float,
                               ood_distance: float,
                               limitations_triggered: List[str]
                               ) -> str:
        n_active = sum(1 for v in safeguards.values() if v)
        n_total = len(safeguards)
        
        res = f"{n_active}/{n_total} safeguards active. "
        
        has_warning = False
        
        if ece > 0.25:
            res += "Calibration warning: confidence estimates are unreliable. "
            has_warning = True
            
        if ood_distance > 3.0:
            res += "Out-of-distribution warning: current observations are outside the training envelope. "
            has_warning = True
            
        if limitations_triggered:
            res += f"Active limitations: {', '.join(limitations_triggered)}. "
            has_warning = True
            
        if not has_warning:
            res += "All monitored safety conditions are nominal."
            
        return res.strip()
