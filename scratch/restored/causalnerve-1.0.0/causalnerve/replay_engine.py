import json
import time
import sys
from causalnerve_observatory import CausalNerveObservatory

class ReplayEngine:
    """Loads and replays causal telemetry logs."""
    
    def __init__(self, log_path: str, speed_multiplier: float = 1.0):
        self.log_path = log_path
        self.speed_multiplier = speed_multiplier
        self.obs = CausalNerveObservatory(port=8766, scenario="replay", auto_open=True)
        
    def run(self):
        print(f"Starting Replay Engine from {self.log_path}")
        self.obs.start()
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Could not find telemetry log {self.log_path}")
            return
            
        print(f"Loaded {len(lines)} telemetry frames.")
        
        last_time = None
        for i, line in enumerate(lines):
            try:
                frame = json.loads(line.strip())
                cycle = frame.get('cycle', i)
                
                # Calculate sleep to maintain original pace (or multiplied)
                # In demo it's roughly 0.05 seconds per cycle, but we'll use a fixed interval for smooth replay
                time.sleep(0.05 / self.speed_multiplier)
                
                self.obs.update(cycle, frame)
                sys.stdout.write(f"\rReplaying cycle {cycle}/{len(lines)}...")
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                continue
                
        print("\nReplay complete. Dashboard remains active.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down replay.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CausalNerve Telemetry Replay")
    parser.add_argument("log_path", help="Path to telemetry .jsonl file")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    args = parser.parse_args()
    
    engine = ReplayEngine(args.log_path, args.speed)
    engine.run()
