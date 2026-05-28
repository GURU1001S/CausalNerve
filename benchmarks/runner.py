import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict

from benchmarks.real_baselines.models import MockVarLingam, MockDynoTears, MockPCMCI
from benchmarks.evaluation import SharedEvaluationProtocol
from benchmarks.bench_statistics import StatisticalPipeline

class BenchmarkRunner:
    """Executes identical evaluation protocols across all CausalNerve baselines."""
    
    def __init__(self, n_nodes: int = 14, time_steps: int = 500, trials: int = 3):
        self.n_nodes = n_nodes
        self.time_steps = time_steps
        self.trials = trials
        self.models = {
            "VARLiNGAM": MockVarLingam,
            "DYNOTEARS": MockDynoTears,
            "PCMCI": MockPCMCI
        }
        self.results = defaultdict(lambda: defaultdict(list))
        self.failures = []

    def _generate_synthetic_data(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates mock non-stationary time-series data and a ground-truth DAG."""
        np.random.seed(seed)
        data = np.random.randn(self.time_steps, self.n_nodes)
        
        # Ground truth DAG (sparse)
        true_adj = np.random.rand(self.n_nodes, self.n_nodes) < 0.1
        np.fill_diagonal(true_adj, 0)
        true_adj = true_adj.astype(float)
        
        # Simulate some dependencies
        for t in range(1, self.time_steps):
            data[t] += data[t-1] @ true_adj * 0.5
            
        return data, true_adj

    def run(self):
        print(f"Starting rigorous causal benchmark. {self.trials} trials, {self.time_steps} steps.")
        
        for trial in range(self.trials):
            seed = 42 + trial
            data, true_adj = self._generate_synthetic_data(seed)
            
            for name, ModelClass in self.models.items():
                try:
                    model = ModelClass(seed=seed)
                    model.fit(data)
                    
                    pred_adj = model.predict_structure()
                    conf = model.confidence_scores()
                    
                    metrics = SharedEvaluationProtocol.evaluate(
                        pred_adj, conf, true_adj, model.runtime_sec
                    )
                    
                    for k, v in metrics.items():
                        self.results[name][k].append(v)
                        
                except Exception as e:
                    self.failures.append({
                        "model": name,
                        "seed": seed,
                        "error": str(e)
                    })

    def generate_report(self, filepath: str = "BENCHMARK_REPORT.md"):
        report = []
        report.append("# Causal Discovery Scientific Benchmarks")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Protocol**: {self.trials} trials | {self.time_steps} time-steps | {self.n_nodes} variables\n")
        
        report.append("## Average Performance Metrics")
        report.append("| Model | SHD (↓) | SID (↓) | ECE (↓) | False Surgery Rate (↓) | Runtime (s) |")
        report.append("|-------|---------|---------|---------|------------------------|-------------|")
        
        for name in self.models.keys():
            m = self.results[name]
            if not m:
                report.append(f"| {name} | FAILED | FAILED | FAILED | FAILED | FAILED |")
                continue
            
            shd = np.mean(m["SHD"])
            sid = np.mean(m["SID"])
            ece = np.mean(m["ECE"])
            fsr = np.mean(m["False_Surgery_Rate"])
            rt = np.mean(m["Runtime_sec"])
            
            report.append(f"| {name} | {shd:.2f} | {sid:.2f} | {ece:.3f} | {fsr:.3f} | {rt:.3f} |")
            
        report.append("\n## Statistical Significance Testing")
        report.append("Comparing PCMCI against VARLiNGAM (baseline).")
        
        if "PCMCI" in self.results and "VARLiNGAM" in self.results:
            shd_stats = StatisticalPipeline.compare_models(
                "SHD", self.results["VARLiNGAM"]["SHD"], self.results["PCMCI"]["SHD"]
            )
            report.append(f"- **SHD p-value (Wilcoxon)**: {shd_stats['wilcoxon_p']:.4f}")
            report.append(f"- **Cohen's d**: {shd_stats['cohens_d']:.2f}")
            report.append(f"- **Statistically Significant**: {shd_stats['significant']}")
            
        report.append("\n## Benchmark Failures & Audit")
        if not self.failures:
            report.append("All models completed 100% of trials successfully.")
        else:
            for f in self.failures:
                report.append(f"- `{f['model']}` failed on seed {f['seed']}: {f['error']}")
                
        from benchmarks.leaderboard.report_generator import LeaderboardReportGenerator
        LeaderboardReportGenerator.generate_html(self.results, self.failures, filepath.replace('.md', '.html'))

if __name__ == "__main__":
    runner = BenchmarkRunner(trials=5)
    runner.run()
    runner.generate_report()
