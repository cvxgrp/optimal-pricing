
from data import generate_data
from pricing import solve_ppp

    
profit_data, constraint_data = generate_data(n=1280)
result = solve_ppp(profit_data, constraint_data, method='QMM')

print(f'Profit: {result.profit.round(1)}')
print(f'Prices changes: {result.price_changes.round(4)}')
