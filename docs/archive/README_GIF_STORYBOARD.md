# CausalNerve Viral Demo Scenarios & GIF Storyboard

To make developers and researchers instantly understand the value of CausalNerve, the GitHub README will feature 3 high-quality, auto-playing GIFs (generated via `export_gif.py` and Plotly animations).

These scenarios focus on conveying: **"This graph is alive, it knows when it's wrong, and it fixes itself."**

---

## Demo 1: The "persistent adaptive causal graph" (Structural Healing)
**Visual:** A dark-themed 3D network graph (force-directed layout).
1. **Normal Operation:** The graph breathes. Edges pulse with a soft blue (`#58A6FF`) glow as data flows through them. Nodes are green (`#238636`).
2. **The Anomaly:** A red anomaly wave hits Node 4 (`#F85149`). The node flashes violently.
3. **The Alarm Propagation:** The red anomaly travels down the edge `4 -> 7`. Edge `4 -> 7` turns bright red, thickens, and begins to vibrate (simulating structural leakage).
4. **The Surgery (The "Wow" Moment):** A purple cursor/animation sweeps in. The faulty edge `4 -> 7` is sliced (turns gray `#484F58` and becomes a dashed line). Simultaneously, a new bright purple edge (`#A371F7`) sprouts from `4 -> 9`, bridging the gap.
5. **Recovery:** The red alarms fade out. The graph resumes its soft blue breathing, having successfully adapted to the new physical reality.

## Demo 2: Dual-World Counterfactual Split Screen
**Visual:** A high-tech two-panel dashboard with synchronous timeline scrubbing.
1. **Left Panel (Factual Future):** Shows a chaotic, oscillating time-series trajectory where a cascading failure destroys 4 downstream sensors.
2. **Right Panel (Intervened Future):** Shows the exact same timeline, but an explicit $do(Sensor\_2 = 0)$ mathematical intervention has been applied. The 4 downstream sensors remain perfectly stable.
3. **The Divergence:** The bottom row shows the mathematically computed divergence integral (purple shaded area), dynamically filling up as the timeline scrubs forward, proving the massive monetary/physical value of executing the proposed graph surgery.

## Demo 3: Fleet Structural Epidemiology
**Visual:** A global/fleet map transitioning into a Dynamic Time Warping (DTW) heatmap.
1. **The Match:** The matrix suddenly highlights a diagonal bright yellow "Optimal Alignment Path."
2. **The Warning:** The UI flashes: *"Asset 42 structural trajectory matches Historical Outbreak (Asset 18, 2024)."*
3. **The Prediction:** The UI automatically projects the next structural phase transition, predicting which edges will break 200 cycles into the future.
