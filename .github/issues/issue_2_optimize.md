# Optimization: Vectorize Adjacency Extraction in `CausalNerve.step()`

## Description
During high-frequency streaming (`CausalNerve.step()`), we extract the adjacency matrix edges and weights. Currently, this extraction might rely on standard Python loops or list comprehensions, which can be a bottleneck when the number of nodes `N` is large.

## What to do
1. Locate the edge extraction logic in `causalnerve/core.py` (or wherever the adjacency matrix is unpacked into tuples).
2. Replace the loop with vectorized `torch.nonzero()` or `numpy.argwhere()` operations to quickly extract coordinates and values where the weight exceeds the threshold.

## Why this is a good first issue
It is a strictly scoped performance optimization. You only need to deal with a single function and basic PyTorch/NumPy indexing, without touching the complex theoretical engines.
