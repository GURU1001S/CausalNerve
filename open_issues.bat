@echo off
echo Opening Good First Issues on GitHub...

gh issue create --title "Feature Request: Add 'Mean Absolute Leakage' Metric to Gradio Dashboard" --body-file .github\issues\issue_1_metric.md --label "good first issue"
gh issue create --title "Optimization: Vectorize Adjacency Extraction in CausalNerve.step()" --body-file .github\issues\issue_2_optimize.md --label "good first issue, performance"
gh issue create --title "Feature Request: Graceful Fallback for Plotly WebGL Context Failures" --body-file .github\issues\issue_3_fallback.md --label "good first issue, ui"

echo Issues opened successfully!
