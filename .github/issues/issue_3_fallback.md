# Feature Request: Graceful Fallback for Plotly WebGL Context Failures

## Description
The `causalnerve-observe` dashboard relies heavily on Plotly WebGL (`Scattergl`) to render massive causal graphs quickly. However, in environments with limited GPU resources or strict browser security policies, WebGL context creation can fail, causing the dashboard graph to go blank.

## What to do
1. Add a fallback mechanism in `causalnerve_observe/dashboard.py` (or where the graph trace is built).
2. Catch WebGL-related errors or provide a UI toggle to switch between `Scattergl` and standard `Scatter` traces.
3. (Optional) Provide a simple `matplotlib` image generation fallback if Plotly fails entirely.

## Why this is a good first issue
This is a robustification task that improves the accessibility of the framework across different devices. It involves conditional logic and interacting with Plotly's graph objects.
