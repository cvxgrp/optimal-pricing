# Optimal Product Pricing

This package contains code for solving product pricing problems (PPPs) as described in our
[note on optimal product pricing](https://stanford.edu/~boyd/papers/optimal_pricing.html), equation (6).
Three methods for solving a PPP are supported:
quadratic minorization maximization (QMM), convex concave procedure (CCP), and nonlinear programming (NLP).
For mathematical background, please refer to our manuscript.

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

To reproduce the figures from the manuscript, run

```bash
python figure2.py
python figure3.py
```

## Dependencies

- Python 3.12
- NumPy
- SciPy
- CVXPY
- Matplotlib (for figure generation)
