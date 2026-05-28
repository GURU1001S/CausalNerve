import sys
import os

# Add repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalnerve.api import CausalNerve

def main():
 print("=" * 70)
 print(" CausalNerve - Phase 2: Live Graph Visualizer (NASA FD001)")
 print("=" * 70)
 
 # 21 sensors in CMAPSS dataset
 nerve = CausalNerve(nodes=21, state_dim=64)
 nerve.alarm_threshold = 0.05
 
 print("\n[STREAM] Engaging CausalNerve.visualize_live()...")
 print("[STREAM] Capturing causal telemetry and generating HTML frames...")
 
 # Run the live visualization
 # We use a fast sleep_factor for the demo script so it doesn't take 5 minutes
 # It will record frames per cycle and compile them into a continuous HTML render
 vis = nerve.visualize_live(engine_id=1, realtime=True, sleep_factor=0.01, output_file="live_nasa_evolution.html")
 
 print("\n" + "=" * 70)
 print("[SUCCESS] Live streaming visualization complete!")
 print("[SUCCESS] The continuous render has been exported.")
 print("To view the animation:")
 print(" -> Open 'live_nasa_evolution.html' in any web browser.")
 print(" -> Click the 'Play Stream' button at the bottom.")
 print("=" * 70)

if __name__ == "__main__":
 main()
