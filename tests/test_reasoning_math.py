"""
tests/test_reasoning_math.py
============================
Validates that CausalNerve reasoning methods use real causal inference math.

Tests:
  1. Interventions alter descendants only (isolation guarantee)
  2. Causal influence propagates correctly through DAG
  3. Counterfactual divergence increases after intervention
  4. Root-cause traces reference real graph paths
  5. Explanations contain real node names and contribution scores
"""

import pytest
import numpy as np
from causalnerve.interventions.intervention import CausalGraph, InterventionEngine
from causalnerve.interventions.counterfactual import CounterfactualEngine
from causalnerve.interventions.trace import CausalTracer


# ───────────────────────────────────────────────────────────
#  Fixtures: Build the NASA C-MAPSS aerospace DAG
# ───────────────────────────────────────────────────────────

AERO_NODES = {
    0: "Fan", 1: "LPC", 2: "HPC", 3: "Combustor", 4: "HPT",
    5: "LPT", 6: "H.Spool", 7: "L.Spool", 8: "P.Bank",
    9: "Cooling", 10: "Bypass", 11: "Fuel", 12: "Snsr.A", 13: "Snsr.B"
}

AERO_EDGES = [
    (11, 3), (3, 4), (4, 2), (4, 6), (6, 2), (2, 1),
    (5, 7), (7, 0), (9, 4), (10, 1), (4, 12), (3, 12)
]


@pytest.fixture
def aero_graph():
    return CausalGraph(n_nodes=14, edges=AERO_EDGES, node_labels=AERO_NODES)


@pytest.fixture
def ie():
    return InterventionEngine()


@pytest.fixture
def cf(ie):
    return CounterfactualEngine(ie)


@pytest.fixture
def tracer():
    return CausalTracer(temporal_decay=0.85)


# ───────────────────────────────────────────────────────────
#  1. GRAPH STRUCTURE TESTS
# ───────────────────────────────────────────────────────────

class TestGraphStructure:
    def test_children(self, aero_graph):
        # Combustor (3) -> HPT (4), Snsr.A (12)
        children = aero_graph.children(3)
        assert 4 in children
        assert 12 in children

    def test_parents(self, aero_graph):
        # HPT (4) parents: Combustor (3), Cooling (9)
        parents = aero_graph.parents(4)
        assert 3 in parents
        assert 9 in parents

    def test_descendants(self, aero_graph):
        # Fuel (11) -> Combustor -> HPT -> {HPC, H.Spool, Snsr.A} -> ...
        desc = aero_graph.descendants(11)
        assert 3 in desc   # Combustor
        assert 4 in desc   # HPT
        assert 2 in desc   # HPC
        assert 12 in desc  # Snsr.A
        # Fan, LPT, L.Spool should NOT be descendants of Fuel
        assert 0 not in desc  # Fan
        assert 5 not in desc  # LPT

    def test_topological_order(self, aero_graph):
        topo = aero_graph.topological_order()
        assert len(topo) == 14
        # Parents must appear before children
        for (src, dst) in AERO_EDGES:
            assert topo.index(src) < topo.index(dst), \
                f"{AERO_NODES[src]} should come before {AERO_NODES[dst]} in topo order"


# ───────────────────────────────────────────────────────────
#  2. INTERVENTION TESTS (Pearl do-calculus)
# ───────────────────────────────────────────────────────────

class TestIntervention:
    def test_do_alters_intervened_node(self, aero_graph, ie):
        states = np.full(14, 0.5)
        result = ie.do(aero_graph, states, node=4, value=0.9)  # do(HPT = 0.9)
        
        assert result.post_intervention_states[4] == pytest.approx(0.9)
        assert result.intervention_value == 0.9

    def test_do_propagates_to_descendants(self, aero_graph, ie):
        states = np.full(14, 0.5)
        result = ie.do(aero_graph, states, node=4, value=0.9)  # do(HPT = 0.9)
        
        # HPC (2) is a descendant of HPT (4) via edge 4->2
        # Its state should differ from the original 0.5
        desc = aero_graph.descendants(4)
        assert 2 in desc
        assert result.post_intervention_states[2] != pytest.approx(0.5, abs=1e-6)

    def test_do_does_not_alter_non_descendants(self, aero_graph, ie):
        """The causal isolation guarantee: non-descendants are untouched."""
        states = np.full(14, 0.5)
        result = ie.do(aero_graph, states, node=4, value=0.9)
        
        non_desc = result.non_descendants_unchanged
        for nd in non_desc:
            assert result.post_intervention_states[nd] == pytest.approx(0.5, abs=1e-8), \
                f"Non-descendant {AERO_NODES[nd]} was altered! Value: {result.post_intervention_states[nd]}"

    def test_isolation_report_is_clean(self, aero_graph, ie):
        states = np.full(14, 0.5)
        result = ie.do(aero_graph, states, node=11, value=1.0)  # do(Fuel = 1.0)
        
        assert result.isolation_report.is_isolated is True
        assert len(result.isolation_report.violations) == 0

    def test_propagation_log_is_nonempty(self, aero_graph, ie):
        states = np.full(14, 0.5)
        result = ie.do(aero_graph, states, node=11, value=1.0)
        
        # Should have at least: sever + clamp + propagation steps
        assert len(result.propagation_log) >= 2
        steps = [e["step"] for e in result.propagation_log]
        assert "sever" in steps
        assert "clamp" in steps


# ───────────────────────────────────────────────────────────
#  3. COUNTERFACTUAL TESTS (Dual-world divergence)
# ───────────────────────────────────────────────────────────

class TestCounterfactual:
    def test_divergence_is_positive(self, aero_graph, cf):
        """Intervening should produce nonzero divergence between worlds."""
        result = cf.simulate(aero_graph, intervention={4: 0.9}, horizon=30)
        assert result.cumulative_divergence > 0

    def test_divergence_increases_with_stronger_intervention(self, aero_graph, cf):
        """A stronger intervention should produce more divergence."""
        states = np.full(14, 0.5)
        
        r_small = cf.simulate(aero_graph, {4: 0.55}, initial_states=states.copy(), horizon=30)
        r_large = cf.simulate(aero_graph, {4: 0.95}, initial_states=states.copy(), horizon=30)
        
        assert r_large.cumulative_divergence > r_small.cumulative_divergence

    def test_affected_nodes_are_descendants(self, aero_graph, cf):
        """Only descendants and the target should be flagged as affected."""
        result = cf.simulate(aero_graph, intervention={11: 1.0}, horizon=30)
        
        desc = aero_graph.descendants(11) | {11}
        for n in result.affected_nodes:
            assert n in desc, f"Node {AERO_NODES[n]} flagged as affected but is not a descendant of Fuel"

    def test_trajectories_have_correct_shape(self, aero_graph, cf):
        result = cf.simulate(aero_graph, intervention={4: 0.9}, horizon=30)
        
        assert result.world_0_trajectory.shape == (30, 14)
        assert result.world_1_trajectory.shape == (30, 14)
        assert result.divergence.shape == (30,)
        assert result.per_node_divergence.shape == (30, 14)

    def test_intervention_is_persistent(self, aero_graph, cf):
        """The intervened node should stay clamped across all timesteps."""
        result = cf.simulate(aero_graph, intervention={4: 0.9}, horizon=30)
        
        for t in range(30):
            assert result.world_1_trajectory[t, 4] == pytest.approx(0.9)


# ───────────────────────────────────────────────────────────
#  4. ROOT-CAUSE TRACING TESTS
# ───────────────────────────────────────────────────────────

class TestRootCauseTracing:
    def test_trace_finds_parents(self, aero_graph, tracer):
        """Tracing HPT should find Combustor and Cooling as immediate parents."""
        result = tracer.trace(aero_graph, anomalous_node=4)
        
        # The contribution percentages should include Combustor (3) and Cooling (9)
        assert 3 in result.contribution_percentages or 9 in result.contribution_percentages

    def test_trace_finds_deep_precursor(self, aero_graph, tracer):
        """Tracing HPC (2) should trace back through HPT and further to Fuel."""
        result = tracer.trace(aero_graph, anomalous_node=2)
        
        # Fuel (11) is the deepest root cause via 11->3->4->2
        all_roots = [c.path[0] for c in result.ranked_causes]
        assert 11 in all_roots or 9 in all_roots  # Either Fuel or Cooling

    def test_ranked_causes_are_sorted(self, aero_graph, tracer):
        result = tracer.trace(aero_graph, anomalous_node=2)
        
        scores = [c.influence_score for c in result.ranked_causes]
        assert scores == sorted(scores, reverse=True)

    def test_paths_reference_real_edges(self, aero_graph, tracer):
        """Every consecutive pair in a causal chain must be a real edge in the DAG."""
        result = tracer.trace(aero_graph, anomalous_node=1)  # LPC
        
        edge_set = set(AERO_EDGES)
        for chain in result.ranked_causes:
            for i in range(len(chain.path) - 1):
                edge = (chain.path[i], chain.path[i + 1])
                assert edge in edge_set, \
                    f"Path contains non-existent edge {AERO_NODES[edge[0]]} -> {AERO_NODES[edge[1]]}"

    def test_confidence_is_bounded(self, aero_graph, tracer):
        result = tracer.trace(aero_graph, anomalous_node=4)
        assert 0.0 <= result.confidence <= 1.0


# ───────────────────────────────────────────────────────────
#  5. SDK INTEGRATION TESTS
# ───────────────────────────────────────────────────────────

class TestSDKIntegration:
    def test_why_returns_real_data(self):
        from causalnerve.api import CausalNerve
        engine = CausalNerve.from_preset("aerospace")
        
        result = engine.why("HPC")
        assert "parents" in result
        assert len(result["parents"]) > 0
        assert "ranked_chains" in result
        assert len(result["ranked_chains"]) > 0
        assert "confidence" in result
        assert result["confidence"] > 0

    def test_what_if_returns_divergence(self):
        from causalnerve.api import CausalNerve
        engine = CausalNerve.from_preset("aerospace")
        
        result = engine.what_if("HPT", 0.2)
        assert "cumulative_divergence" in result
        assert result["cumulative_divergence"] > 0
        assert "affected_nodes" in result
        assert "confidence" in result

    def test_do_mutates_state(self):
        from causalnerve.api import CausalNerve
        engine = CausalNerve.from_preset("aerospace")
        
        result = engine.do("Fuel", 1.0)
        assert result["status"] == "success"
        assert result["new_value"] == 1.0
        assert len(result["descendants_affected"]) > 0
        assert result["isolation_verified"] is True

    def test_run_counterfactual_produces_divergence(self):
        from causalnerve.api import CausalNerve
        engine = CausalNerve.from_preset("aerospace")
        
        result = engine.run_counterfactual({"node": "HPT", "value": 0.5})
        assert result["divergence"] > 0
        assert "leakage_reduction" in result
