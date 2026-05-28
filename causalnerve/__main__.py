import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="CausalNerve Health Check")
    parser.parse_args()
    
    print("--- CausalNerve System Health Check ---\n")
    
    try:
        import causalnerve
        print(f"[OK] Package Version: {causalnerve.__version__}")
    except ImportError:
        print("[FAIL] CausalNerve package not found.")
        sys.exit(1)
        
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"     CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("[FAIL] PyTorch not found.")
        
    modules = ["numpy", "pandas", "networkx", "scipy", "sklearn", "fastapi", "dash"]
    for mod in modules:
        try:
            import importlib
            importlib.import_module(mod)
            print(f"[OK] {mod}: Installed")
        except ImportError:
            print(f"[FAIL] {mod}: Missing")

    print("\nSystem ready. Try running `causalnerve-demo` or `causalnerve-observatory`.")

if __name__ == "__main__":
    main()
