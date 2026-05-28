import sys
import logging
from causalnerve import CausalNerve
from causalnerve.datasets import SyntheticStreamGenerator
from causalnerve.benchmarks.msrb import MSRBSuite

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    mode = "quick" if len(sys.argv) == 1 else sys.argv[1]
    
    suite = MSRBSuite()
    nerve_engine = CausalNerve(nodes=6, state_dim=32)
    
    # Factory that returns a clean data iterator for testing
    def stream_factory(seed):
        return SyntheticStreamGenerator.stable(n_cycles=500, n_nodes=6)
        
    print(f"Running MSRB {mode} suite...")
    
    if mode == "quick":
        suite.run_quick_suite(nerve_engine, stream_factory, n_seeds=5)
    elif mode == "full":
        suite.run_full_suite(nerve_engine, stream_factory, n_seeds=20)
    else:
        print(f"Unknown mode: {mode}. Use 'quick' or 'full'.")
        sys.exit(1)
        
    print("MSRB suite execution complete!")

if __name__ == "__main__":
    main()
