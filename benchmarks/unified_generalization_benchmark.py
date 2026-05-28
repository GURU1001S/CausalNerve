import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalnerve.api import CausalNerve
from causalnerve.plugins.registry import PluginRegistry
from causalnerve.plugins.eeg_plugin import EEGDomainPlugin

# Register EEG domain
PluginRegistry.register(EEGDomainPlugin())

def get_nerve(domain):
    return CausalNerve.from_preset(domain)

def apply_stress(telemetry, regime, t, n_nodes):
    val = telemetry.copy()
    
    if regime == "Gaussian Noise":
        val += np.random.randn(n_nodes) * 0.2
        
    elif regime == "Correlated Noise":
        cov = np.ones((n_nodes, n_nodes)) * 0.1
        np.fill_diagonal(cov, 0.3)
        val += np.random.multivariate_normal(np.zeros(n_nodes), cov)
        
    elif regime == "Packet Dropout":
        if np.random.rand() < 0.2:
            val = np.zeros(n_nodes) # Simulate missing data
            
    elif regime == "Hidden Confounder":
        # Node 0 and Node 1 driven by unobserved Z
        z = np.sin(t / 5.0) * 0.8
        val[0] += z
        val[1] += z
        
    elif regime == "Delayed Effects":
        # Target node gets delayed signal
        pass # Too complex for simple instantaneous mapping without state buffer
        
    elif regime == "Regime Shift":
        if t > 25:
            val += 1.5
            
    elif regime == "Partial Observability":
        mask = np.random.rand(n_nodes) > 0.3
        val[mask] = 0.0
        
    elif regime == "Adversarial Corruption":
        val[np.random.randint(0, n_nodes)] += 2.0
        
    elif regime == "Random Rewiring":
        if t % 5 == 0:
            val = np.random.randn(n_nodes) * 1.5
            
    elif regime == "Sampling Rate Degradation":
        # Interpolated hold
        pass 
        
    return val

def run_stress_test(domain, regime, cycles=50):
    nerve = get_nerve(domain)
    n_nodes = nerve.graph.n_nodes
    
    # Generate baseline data
    base_telemetry = np.random.rand(cycles, n_nodes)
    
    leakages = []
    alarms_fired = 0
    revisions = 0
    
    for t in range(cycles):
        telemetry = base_telemetry[t]
        
        # Inject structural fault midway
        if t >= cycles // 2:
            if domain == "aerospace":
                telemetry[4] += telemetry[3] * 1.5 # HPT -> Combustor
            else:
                telemetry[9] += telemetry[8] * 1.5 # Cz -> C3
                
        # Apply stress
        stressed = apply_stress(telemetry, regime, t, n_nodes)
        
        # Run Watch
        result = nerve.watch(stressed)
        
        leakages.append(result.leakage)
        alarms_fired += len(result.alarms)
        revisions += len(result.revisions)
        
    # Metrics
    leakage_integral = np.sum(leakages)
    false_alarms = alarms_fired if regime != "Baseline" else alarms_fired
    
    return {
        "Leakage_Integral": leakage_integral,
        "Alarms": alarms_fired,
        "Revisions": revisions,
        "Mean_Leakage": np.mean(leakages)
    }

def main():
    print("[*] Starting Unified Generalization Benchmark")
    os.makedirs("results", exist_ok=True)
    
    domains = ["aerospace", "eeg"]
    regimes = [
        "Baseline", "Gaussian Noise", "Correlated Noise", "Packet Dropout",
        "Hidden Confounder", "Regime Shift", "Partial Observability",
        "Adversarial Corruption", "Random Rewiring"
    ]
    
    results = {d: {} for d in domains}
    
    for domain in domains:
        print(f"\n[+] Testing Domain: {domain.upper()}")
        for regime in regimes:
            res = run_stress_test(domain, regime)
            results[domain][regime] = res
            print(f"  - {regime}: Mean Leakage={res['Mean_Leakage']:.4f}, Alarms={res['Alarms']}")
            
    # Compute Cross-Domain Consistency
    print("\n[*] Computing Cross-Domain Consistency...")
    # Calculate correlation of mean leakages across regimes between domains
    turbo_leaks = [results["aerospace"][r]["Mean_Leakage"] for r in regimes]
    eeg_leaks = [results["eeg"][r]["Mean_Leakage"] for r in regimes]
    
    corr = np.corrcoef(turbo_leaks, eeg_leaks)[0, 1]
    print(f"    -> Robustness Correlation (Aerospace vs EEG): {corr:.3f}")
    
    # Visualizations
    print("[*] Generating Visual Outputs...")
    
    # 1. Generalization Heatmap
    heatmap_data = np.array([turbo_leaks, eeg_leaks])
    plt.figure(figsize=(10, 4))
    sns.heatmap(heatmap_data, annot=True, xticklabels=regimes, yticklabels=domains, cmap="viridis")
    plt.title("Stress Response Heatmap (Mean Leakage)")
    plt.tight_layout()
    plt.savefig("results/generalization_heatmap.png")
    plt.close()
    
    # 2. Domain Robustness Radar
    angles = np.linspace(0, 2 * np.pi, len(regimes), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for domain in domains:
        vals = [results[domain][r]["Mean_Leakage"] for r in regimes]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=domain.upper())
        ax.fill(angles, vals, alpha=0.25)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(regimes, size=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Domain Robustness Radar")
    plt.tight_layout()
    plt.savefig("results/domain_robustness_radar.png")
    plt.close()
    
    # 3. Stress Response Curves
    # (Since this is aggregated, we just plot the leakage comparison bar chart)
    plt.figure(figsize=(12, 5))
    x = np.arange(len(regimes))
    width = 0.35
    plt.bar(x - width/2, turbo_leaks, width, label='Aerospace')
    plt.bar(x + width/2, eeg_leaks, width, label='EEG')
    plt.ylabel('Mean Leakage')
    plt.title('Stress Response Curves (Aggregated)')
    plt.xticks(x, regimes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/stress_response_curves.png")
    plt.close()
    
    print("[+] Visualizations saved to results/ directory.")

if __name__ == "__main__":
    main()
