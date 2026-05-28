import pytest
import os
from benchmarks.leaderboard.aggregator import ResultsAggregator
from benchmarks.leaderboard.visualizer import BenchmarkVisualizer
from benchmarks.leaderboard.report_generator import LeaderboardReportGenerator

def test_aggregator():
    raw_results = {
        "ModelA": {
            "SHD": [2.0, 4.0, 3.0],
            "Runtime_sec": [1.0, 1.2, 1.1]
        }
    }
    
    aggregated = ResultsAggregator.aggregate(raw_results)
    
    assert "ModelA" in aggregated
    assert aggregated["ModelA"]["SHD"]["mean"] == 3.0
    assert aggregated["ModelA"]["SHD"]["std"] == 1.0

def test_visualizer():
    aggregated = {
        "ModelA": {
            "SHD": {"mean": 2.5, "ci_upper": 3.0}
        }
    }
    
    svg = BenchmarkVisualizer.generate_shd_plot(aggregated)
    assert "<svg" in svg
    assert "ModelA" in svg
    assert "2.5" in svg
    
def test_report_generator():
    raw_results = {
        "ModelA": {
            "SHD": [2.0, 4.0],
            "Runtime_sec": [1.0, 1.2]
        }
    }
    failures = [{"model": "ModelB", "seed": 42, "error": "Crash"}]
    
    filepath = "TEST_LEADERBOARD.html"
    LeaderboardReportGenerator.generate_html(raw_results, failures, filepath)
    
    assert os.path.exists(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()
        assert "ModelA" in html
        assert "ModelB" in html # failure
        
    os.remove(filepath)
