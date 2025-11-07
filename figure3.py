
import time
import numpy as np
import matplotlib.pyplot as plt

from data import generate_data
from optimization import solve_ppp


n_ = [20, 40, 80, 160, 320, 640, 1280, 2560]


times = np.zeros((3, len(n_)))

for i, n in enumerate(n_):
    
    profit_data, constraint_data = generate_data(n=n)
    
    t = time.time()
    solve_ppp(profit_data, constraint_data, method='CCP')
    times[0, i] = time.time() - t
    
    t = time.time()
    solve_ppp(profit_data, constraint_data, method='QMM')
    times[1, i] = time.time() - t
    
    t = time.time()
    solve_ppp(profit_data, constraint_data, method='NLP')
    times[2, i] = time.time() - t
    

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(8, 4))
plt.loglog(n_, times[2], 'mo-', label='NLP')
plt.loglog(n_, times[0], 'bo-', label='CCP')
plt.loglog(n_, times[1], 'go-', label='QMM')
plt.xlim(10, 5000)
plt.xlabel('$n$')
plt.ylabel('Time [s]')
plt.legend()
plt.show()
