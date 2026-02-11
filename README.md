# Optimal Product Pricing

This package contains code for solving product pricing problems (PPPs) as described in our
[note on optimal product pricing](https://stanford.edu/~boyd/papers/optimal_pricing.html)
(equation (3)).
Three methods for solving a PPP are provided:
quadratic minorization maximization (QMM), convex concave procedure (CCP), and nonlinear programming (NLP).
For mathematical background, please refer to the [manuscript](https://stanford.edu/~boyd/papers/optimal_pricing.html).

## Dependencies

This code depends on the Python packages `CVXPY`, `SciPy`, and `Matplotlib` (for figure generation).
If you wish to use NLP, please install `CVXPY` from sources from [this fork](https://github.com/cvxgrp/DNLP),
until CVXPY with support for NLP is released.

## Example

Here's a simple example for optimizing the prices of 1280 products with QMM:

```python
from data import generate_data
from optimization import solve_ppp


# Generate sample data for 1280 products
profit_data, constraint_data = generate_data(n=1280)

# Solve using QMM
result = solve_ppp(profit_data, constraint_data, method='QMM')

# Inspect final profit and price changes
print(result.profit)
print(result.price_changes)
```

## Results in manuscript

To reproduce the figures in the manuscript, run:

```bash
python figure2.py
python figure3.py
```
