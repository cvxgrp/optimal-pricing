
import numpy as np
import scipy.sparse as sparse
from typing import Tuple

from pricing import ProfitData, ConstraintData


def construct_elasticity(n: int) -> sparse.csr_matrix:
    """Constructs a sparse elasticity matrix with random entries.
    
    Args:
        n (int): Size of the square matrix
        
    Returns:
        scipy.sparse.csr_matrix: Sparse elasticity matrix
    """
    
    blocks = n // 10

    E = sparse.lil_matrix((n, n))

    for i in range(blocks):
        r = slice(i*10, (i+1)*10)
        E[r, r] = np.random.uniform(-0.05, 0.05, (10, 10))

    E.setdiag(np.random.uniform(-3.0, -1.0, n))

    return E.tocsr()


def generate_data(n: int, seed: int=1) -> Tuple[ProfitData, ConstraintData]:
    """Generate synthetic data for profit optimization problem.
    
    Args:
        n (int): Number of products
        seed (int, optional): Random seed. Defaults to 1.
        
    Returns:
        Tuple[ProfitData, ConstraintData]: Profit and constraint data
    """
    
    np.random.seed(seed)
    
    m = n // 5

    r_nom = 1 + 4 * np.random.rand(n)
    kappa_nom = 0.9 * r_nom
    elasticity = construct_elasticity(n)
    profit_data = ProfitData(r_nom=r_nom, kappa_nom=kappa_nom, elasticity=elasticity)
    
    pi_min, pi_max = np.log(0.8) * np.ones(n), np.log(1.2) * np.ones(n)
    C = np.random.randn(n, m)
    constraint_data = ConstraintData(pi_min=pi_min, pi_max=pi_max, C=C)
    
    return profit_data, constraint_data
