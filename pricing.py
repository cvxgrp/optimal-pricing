
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import List


@dataclass
class ProfitData:
    r_nom: np.ndarray
    kappa_nom: np.ndarray
    elasticity: np.ndarray
    
    
@dataclass
class ConstraintData:
    pi_min: np.ndarray = None
    pi_max: np.ndarray = None
    delta_min: np.ndarray = None
    delta_max: np.ndarray = None
    C: np.ndarray = None
    A: np.ndarray = None
    b: np.ndarray = None
    F: np.ndarray = None
    g: np.ndarray = None


@dataclass
class PricingResult:
    profit: float
    price_changes: np.ndarray
    demand_changes: np.ndarray
    policy_parameters: np.ndarray = None
   
    
def _profit(profit_data: ProfitData, pi_value: np.ndarray) -> float:
    """TODO: Add docstring."""
    return profit_data.r_nom @ np.exp(profit_data.elasticity @ pi_value + pi_value) \
            - profit_data.kappa_nom @ np.exp(profit_data.elasticity @ pi_value)
    
    
def _qmm(profit_data: ProfitData, constraint_data: ConstraintData,
         cvxpy_constraints: List, pi: cp.Variable, delta: cp.Variable,
         theta: cp.Variable=None, tol: float=1e-3) -> PricingResult:
    """TODO: Add docstring."""
    
    n = len(profit_data.r_nom)
    rscaled = cp.CallbackParam(lambda: profit_data.r_nom * np.exp(profit_data.elasticity @ pi.value + pi.value), n)
    alpha = cp.Parameter(n)

    knom_slope = cp.CallbackParam(lambda: profit_data.kappa_nom * np.exp(delta.value) * (1 - alpha.value * delta.value), n)
    knom_curv = cp.CallbackParam(lambda: profit_data.kappa_nom * np.exp(delta.value) * alpha.value / 2, n, nonneg=True)

    obj = rscaled @ (delta + pi) - knom_slope @ delta - knom_curv @ delta**2
    qp = cp.Problem(cp.Maximize(obj), cvxpy_constraints)
    
    delta_max = np.maximum(profit_data.elasticity.toarray(), 0) @ constraint_data.pi_max \
                - np.maximum(-profit_data.elasticity.toarray(), 0) @ constraint_data.pi_min
    
    def _get_alpha(delta_val):
        b = delta_max - delta_val
        return 2 * (np.exp(b) - b - 1) / (b ** 2)

    profits = [_profit(profit_data, np.zeros(n))]

    pi.value = np.zeros(n)
    delta.value = np.zeros(n)
    while True:
        alpha.value = _get_alpha(delta.value)
        qp.solve(solver=cp.OSQP, verbose=False)
        profits.append(_profit(profit_data, pi.value))
        if profits[-1] / profits[-2] - 1 < tol: break
        
    return PricingResult(profit=profits[-1], price_changes=np.exp(pi.value), demand_changes=np.exp(delta.value),
                         policy_parameters=theta.value if theta is not None else None)


def _ccp(profit_data: ProfitData, cvxpy_constraints: List, pi: cp.Variable, delta: cp.Variable,
         theta: cp.Variable=None, tol: float=1e-3) -> PricingResult:
    """TODO: Add docstring."""
    
    n = len(profit_data.r_nom)
    rscaled = cp.CallbackParam(lambda: profit_data.r_nom * np.exp(profit_data.elasticity @ pi.value + pi.value), n)

    obj = rscaled @ (delta + pi) - profit_data.kappa_nom @ cp.exp(delta)
    prob = cp.Problem(cp.Maximize(obj), cvxpy_constraints)

    profits = [_profit(profit_data, np.zeros(n))]
    pi.value = np.zeros(n)
    while True:
        prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
        profits.append(_profit(profit_data, pi.value))
        if profits[-1] / profits[-2] - 1 < tol: break
        
    return PricingResult(profit=profits[-1], price_changes=np.exp(pi.value), demand_changes=np.exp(delta.value),
                         policy_parameters=theta.value if theta is not None else None)


def _nlp(profit_data: ProfitData, cvxpy_constraints: List, pi: cp.Variable, delta: cp.Variable,
         theta: cp.Variable=None, tol: float=1e-3) -> PricingResult:
    """TODO: Add docstring."""
    
    n = len(profit_data.r_nom)
    obj = profit_data.r_nom @ cp.exp(delta + pi) - profit_data.kappa_nom @ cp.exp(delta)
    nlp = cp.Problem(cp.Maximize(obj), cvxpy_constraints)
                
    pi.value = np.zeros(n)
    delta.value = np.zeros(n)
            
    nlp.solve(solver=cp.IPOPT, nlp=True, verbose=False, derivative_test='none', tol=tol)
    
    return PricingResult(profit=_profit(profit_data, pi.value),
                         price_changes=np.exp(pi.value), demand_changes=np.exp(delta.value),
                         policy_parameters=theta.value if theta is not None else None)


def solve_ppp(profit_data: ProfitData, constraint_data: ConstraintData, method: str='QMM', tol: float=1e-3) -> PricingResult:
    """TODO: Add docstring."""
    
    n = len(profit_data.r_nom)
    
    pi = cp.Variable(n, bounds=[constraint_data.pi_min, constraint_data.pi_max])
    delta = cp.Variable(n, bounds=[constraint_data.delta_min, constraint_data.delta_max])
    
    constraints = [delta == profit_data.elasticity @ pi]
        
    if constraint_data.C is not None:
        theta = cp.Variable(constraint_data.C.shape[1])
        constraints += [pi == constraint_data.C @ theta]
    else:
        theta = None
        
    if constraint_data.A is not None and constraint_data.b is not None:
        constraints += [constraint_data.A @ pi == constraint_data.b]
        
    if constraint_data.F is not None and constraint_data.g is not None:
        constraints += [constraint_data.F @ pi <= constraint_data.g]
    
    if method == 'QMM':
        return _qmm(profit_data, constraint_data, constraints, pi, delta, theta, tol)
    elif method == 'CCP':
        return _ccp(profit_data, constraints, pi, delta, theta, tol)
    elif method == 'NLP':
        return _nlp(profit_data, constraints, pi, delta, theta, tol)
    else:
        raise ValueError(f"Unknown solve method: {method}")
