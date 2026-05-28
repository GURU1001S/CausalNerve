import os
import shutil
import glob
import re

base_dir = "causalnerve"

moves = {
    # root
    "sdk.py": "api.py",
    
    # core is mostly unchanged but needs to ensure graph.py exists, maybe it was in streams or events?
    # Actually graph is probably what was `causalnerve/core/graph.py` or maybe it doesn't exist and I shouldn't create it.
    
    # runtime/adaptation
    "adapt/ocgr.py": "runtime/adaptation/ocgr.py",
    "adapt/lyapunov.py": "runtime/adaptation/lyapunov.py",
    "adapt/calibrator.py": "runtime/adaptation/calibrator.py",
    
    # runtime/intervention
    "reason/counterfactual.py": "runtime/intervention/counterfactual.py",
    "reason/intervention.py": "runtime/intervention/intervention.py",
    "reason/trace.py": "runtime/intervention/trace.py",
    
    # runtime/memory
    "fleet/database.py": "runtime/memory/fleet_db.py",
    "memory/recurrence_engine.py": "runtime/memory/recurrence.py",
    "fleet/precognition.py": "runtime/memory/precognition.py",
    
    # runtime/safety
    # Wait, the prompt says "drift_detector.py (move existing drift discriminator here)". I'll check what exists.
    "adapt/causal_sufficiency.py": "runtime/safety/sufficiency.py",
    "adapt/uncertainty_engine.py": "runtime/safety/uncertainty.py",
    
    # reasoning
    "reason/explanation.py": "reasoning/explanation.py",
}

# Add drift detector if we can find it
if os.path.exists(os.path.join(base_dir, "adapt", "live_validation.py")): # maybe?
    pass

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    init_path = os.path.join(os.path.dirname(path), "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, 'w').close()

for src, dst in moves.items():
    src_path = os.path.join(base_dir, src)
    dst_path = os.path.join(base_dir, dst)
    if os.path.exists(src_path):
        ensure_dir(dst_path)
        shutil.move(src_path, dst_path)
        print(f"Moved {src_path} -> {dst_path}")
    else:
        print(f"MISSING: {src_path}")

# Fix imports!
# We must replace old import paths with new ones in all .py files in causalnerve and tests/ and benchmarks/
import_map = {
    "causalnerve.api": "causalnerve.api",
    "causalnerve.runtime.adaptation.ocgr": "causalnerve.runtime.adaptation.ocgr",
    "causalnerve.runtime.adaptation.lyapunov": "causalnerve.runtime.adaptation.lyapunov",
    "causalnerve.runtime.adaptation.calibrator": "causalnerve.runtime.adaptation.calibrator",
    "causalnerve.runtime.intervention.counterfactual": "causalnerve.runtime.intervention.counterfactual",
    "causalnerve.runtime.intervention.intervention": "causalnerve.runtime.intervention.intervention",
    "causalnerve.runtime.intervention.trace": "causalnerve.runtime.intervention.trace",
    "causalnerve.runtime.memory.fleet_db": "causalnerve.runtime.memory.fleet_db",
    "causalnerve.runtime.memory.recurrence": "causalnerve.runtime.memory.recurrence",
    "causalnerve.runtime.memory.precognition": "causalnerve.runtime.memory.precognition",
    "causalnerve.runtime.safety.sufficiency": "causalnerve.runtime.safety.sufficiency",
    "causalnerve.runtime.safety.uncertainty": "causalnerve.runtime.safety.uncertainty",
    "causalnerve.reasoning.explanation": "causalnerve.reasoning.explanation",
}

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = content
            for old_imp, new_imp in import_map.items():
                new_content = new_content.replace(old_imp, new_imp)
                
            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated imports in {path}")
