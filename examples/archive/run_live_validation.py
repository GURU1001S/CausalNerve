import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from causalnerve.api import CausalNerve
from causalnerve.runtime.stream import LiveCMAPSSStream
from causalnerve.runtime.scheduler import LiveMonitoringScheduler

def run():
    print("Initializing CausalNerve...")
    nerve = CausalNerve(nodes=21, state_dim=64)
    # Force some surgeries to be proposed for testing validation
    nerve.alarm_threshold = 0.01 
    
    print("Running stream...")
    # Using the local cached data generated earlier
    stream = LiveCMAPSSStream(engine_id=1, realtime=False)
    scheduler = LiveMonitoringScheduler(nerve, stream)
    
    # We inject some noise to trigger oscillation locks and rollbacks
    def on_cycle(cycle, data, state):
        if cycle % 10 == 0:
            print(f"Cycle {cycle:03d} - Active Surgeries: {len(state.accepted_surgeries)}")
            
    scheduler.run(on_cycle=on_cycle)
    
    os.makedirs("results", exist_ok=True)
    report_path = "results/live_validation_report.md"
    nerve.ocgr.live_validator.generate_report(report_path)
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    run()
