import pytest
import numpy as np
import os
from benchmarks.real_baselines.models import MockVarLingam, MockDynoTears, MockPCMCI
from benchmarks.evaluation import SharedEvaluationProtocol, compute_shd, compute_sid, compute_calibration
from benchmarks.runner import BenchmarkRunner
from benchmarks.statistics import StatisticalPipeline

def test_evaluation_metrics():
    true_adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    pred_adj = np.array([
        [0, 1, 0],
        [1, 0, 0], # false reverse
        [0, 0, 0]
    ])
    
    shd = compute_shd(pred_adj, true_adj)
    assert shd == 2 # 1 missing, 1 false
    
    sid = compute_sid(pred_adj, true_adj)
    assert sid == 2
    
def test_mock_baselines():
    data = np.random.randn(100, 3)
    
    model = MockVarLingam()
    model.fit(data)
    assert model.is_fitted
    assert model.predict_structure().shape == (3, 3)
    assert model.confidence_scores().shape == (3, 3)
    
def test_benchmark_runner():
    runner = BenchmarkRunner(n_nodes=3, time_steps=50, trials=2)
    runner.run()
    
    assert len(runner.results["VARLiNGAM"]["SHD"]) == 2
    
    # Generate report
    runner.generate_report("TEST_REPORT.md")
    assert os.path.exists("TEST_REPORT.md")
    os.remove("TEST_REPORT.md")

def test_statistical_pipeline():
    g1 = [1.0, 1.2, 0.9, 1.1, 1.0]
    g2 = [3.0, 3.2, 2.9, 3.1, 3.0]
    
    stats = StatisticalPipeline.compare_models("SHD", g1, g2)
    assert stats["metric"] == "SHD"
    assert "wilcoxon_p" in stats
