
import numpy as np
import time
from causalnerve.api import CausalNerve
print("[*] Successfully imported CausalNerve")

nerve = CausalNerve.from_preset("aerospace")
print("[*] Loaded aerospace preset")

# Generate random telemetry
telemetry = np.random.rand(14)
nerve.fit(np.random.rand(10, 14))

print("[*] Running Watch Cycle...")
res = nerve.step(telemetry)
print(f"[*] Cycle {res.cycle} completed. Leakage: {res.leakage:.4f}")

print("[*] Running Intervention...")
do_res = nerve.do(3, 1.0)
print(f"[*] do(X=1.0) complete. Isolation verified: {do_res['isolation_verified']}")

nerve.plot_graph("outsider_graph.svg")
print("[*] Graph plotted to outsider_graph.svg")

nerve.export_report("outsider_report.json")
print("[*] Report exported to outsider_report.json")
