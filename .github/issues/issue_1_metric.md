# Feature Request: Add "Mean Absolute Leakage" Metric to Gradio Dashboard

## Description
Currently, the `causalnerve-observe` dashboard displays the real-time Causal Leakage per cycle. We want to add a rolling metric that shows the **Mean Absolute Leakage** over the last 100 cycles to provide a smoother indicator of structural health.

## What to do
1. In `causalnerve_observe/replay.py` (or the equivalent UI logic), maintain a rolling buffer of the last 100 leakage values.
2. Calculate the mean of these values.
3. Add a new `gr.Number()` or `gr.Textbox()` component to the UI layout to display this rolling average.

## Why this is a good first issue
This requires no deep knowledge of causal inference or the math backend—it's purely a UI and state-management task. Perfect for someone getting familiar with `gradio` and the CausalNerve dashboard architecture.
