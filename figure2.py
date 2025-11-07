
import numpy as np
import matplotlib.pyplot as plt

from data import generate_data
from optimization import solve_ppp


n = 1280
    
profit_data, constraint_data = generate_data(n=n)
result_ccp = solve_ppp(profit_data, constraint_data, method='CCP')
result_qmm = solve_ppp(profit_data, constraint_data, method='QMM')


# figure 2a

n_iter = len(result_qmm.profit_iterations) - 1
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(5, 4))
plt.plot(np.arange(n_iter), result_ccp.profit_iterations, 'bo-')
plt.plot(np.arange(n_iter+1), result_qmm.profit_iterations, 'go-', zorder=0)
plt.xticks(np.arange(n_iter+1))
plt.xlabel('Iteration')
plt.ylabel('Profit')
plt.ylim([0.95 * np.min(result_ccp.profit_iterations), 1.05 * np.max(result_ccp.profit_iterations)])
plt.legend(['CCP', 'QMM'])
plt.show()


# figure 2b

xlim = [-50, n + 50]
plt.figure(figsize=(5, 4))
plt.plot(xlim, 100 * (np.exp(constraint_data.pi_min[0])-1) * np.ones(2), 'k--')
plt.plot(xlim, 100 * (np.exp(constraint_data.pi_max[0])-1) * np.ones(2), 'k--')
plt.scatter(np.arange(n), 100 * (result_qmm.price_changes - 1), 15, facecolors='b', edgecolors='b')
plt.xlim(xlim)
plt.xlabel('Price index')
plt.ylabel(r'Price change [\%]')
plt.show()
