
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
    profit_iterations: np.ndarray = None
   
    
def _profit(profit_data: ProfitData, pi: np.ndarray) -> float:
    """Calculate profit for given prices.

    Args:
        profit_data (ProfitData): nominal revenue, nominal cost, and elasticity.
        pi (np.ndarray): Log-price changes.

    Returns:
        float: Profit value.
    """
    delta = profit_data.elasticity @ pi
    return profit_data.r_nom @ np.exp(delta + pi) - profit_data.kappa_nom @ np.exp(delta)
    
    
def _qmm(
    n: int,
    profit_data: ProfitData, constraint_data: ConstraintData,
    cvxpy_constraints: List[cp.Constraint],
    pi: cp.Variable, delta: cp.Variable, theta: cp.Variable=None,
    tol: float=1e-3, maxiter: int=100
) -> PricingResult:
    """Run quadratic minorization maximization.

    Args:
        n (int): Number of products.
        profit_data (ProfitData): nominal revenue, nominal cost, and elasticity.
        constraint_data (ConstraintData): Data containing bounds and linear constraints.
        cvxpy_constraints (List[cp.Constraint]): List of CVXPY constraints.
        pi (cp.Variable): CVXPY Variable for log-price changes (shape (n,)).
        delta (cp.Variable): CVXPY Variable for log-demand changes (shape (n,)).
        theta (cp.Variable | None): Optional CVXPY Variable for policy parameters.
        tol (float): Relative improvement tolerance for convergence.
        maxiter (int): Maximum number of quadratic program solves.

    Returns:
        PricingResult: Final profit, price changes, demand changes and policy parameters.
    """
        
    elasticity = profit_data.elasticity.toarray()
    delta_max = np.maximum(elasticity, 0) @ constraint_data.pi_max \
        - np.maximum(-elasticity, 0) @ constraint_data.pi_min
    delta_max = np.minimum(delta_max, constraint_data.delta_max)
    
    def _get_beta():
        b = delta_max - delta.value
        idx_nz = b > 1e-6
        b_nz = b[idx_nz]
        beta = np.ones(n) / 2
        beta[idx_nz] = 2 * (np.exp(b_nz) - b_nz - 1) / (b_nz ** 2)
        return beta
    
    def _get_rscaled():
        return profit_data.r_nom * np.exp(elasticity @ pi.value + pi.value)
    
    def _get_knom_slope():
        return profit_data.kappa_nom * np.exp(delta.value) * (1 - alpha.value * delta.value)
    
    def _get_knom_curv():
        return profit_data.kappa_nom * np.exp(delta.value) * alpha.value / 2
    
    alpha = cp.CallbackParam(_get_beta, n, nonneg=True)
    rscaled = cp.CallbackParam(_get_rscaled, n)
    knom_slope = cp.CallbackParam(_get_knom_slope, n)
    knom_curv = cp.CallbackParam(_get_knom_curv, n, nonneg=True)

    obj = rscaled @ (delta + pi) - knom_slope @ delta - knom_curv @ delta ** 2
    qp = cp.Problem(cp.Maximize(obj), cvxpy_constraints)
    
    profits = [_profit(profit_data, np.zeros(n))]
    pi.value = np.zeros(n)
    delta.value = np.zeros(n)
    for _ in range(maxiter):
        qp.solve(solver=cp.OSQP, warm_start=True)
        profits.append(_profit(profit_data, pi.value))
        if profits[-1] / profits[-2] - 1 < tol: break
        
    return PricingResult(
        profit=profits[-1],
        profit_iterations=np.array(profits),
        price_changes=np.exp(pi.value),
        demand_changes=np.exp(delta.value),
        policy_parameters=theta.value if theta is not None else None
    )


def _ccp(
    n: int,
    profit_data: ProfitData,
    cvxpy_constraints: List[cp.Constraint],
    pi: cp.Variable, delta: cp.Variable, theta: cp.Variable=None,
    tol: float=1e-3, maxiter: int=100
) -> PricingResult:
    """Run convex-concave procedure.

    Args:
        n (int): number of products.
        profit_data (ProfitData): nominal revenue, nominal cost, and elasticity.
        cvxpy_constraints (List[cp.Constraint]): list of constraints.
        pi (cp.Variable): CVXPY Variable for log-price changes.
        delta (cp.Variable): CVXPY Variable for log-demand changes.
        theta (cp.Variable | None): optional CVXPY Variable for policy parameters.
        tol (float): relative improvement tolerance for convergence.
        maxiter (int): maximum number of convex problem solves.

    Returns:
        PricingResult: Final profit, price changes, demand changes, and policy parameters.
    """
    
    def _get_rscaled():
        return profit_data.r_nom * np.exp(profit_data.elasticity @ pi.value + pi.value)
        
    rscaled = cp.CallbackParam(_get_rscaled, n)
    obj = rscaled @ (delta + pi) - profit_data.kappa_nom @ cp.exp(delta)
    prob = cp.Problem(cp.Maximize(obj), cvxpy_constraints)

    profits = [_profit(profit_data, np.zeros(n))]
    pi.value = np.zeros(n)
    for _ in range(maxiter):
        prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
        profits.append(_profit(profit_data, pi.value))
        if profits[-1] / profits[-2] - 1 < tol: break
        
    return PricingResult(
        profit=profits[-1],
        profit_iterations=np.array(profits),
        price_changes=np.exp(pi.value),
        demand_changes=np.exp(delta.value),
        policy_parameters=theta.value if theta is not None else None
    )


def _nlp(
    n: int,
    profit_data: ProfitData,
    cvxpy_constraints: List[cp.Constraint],
    pi: cp.Variable, delta: cp.Variable, theta: cp.Variable=None,
    tol: float=1e-3
) -> PricingResult:
    """Run nonlinear programming solver.

    Args:
        n (int): Number of products.
        profit_data (ProfitData): nominal revenue, nominal cost, and elasticity.
        cvxpy_constraints (List[cp.Constraint]): list of constraints.
        pi (cp.Variable): CVXPY Variable for log-price changes.
        delta (cp.Variable): CVXPY Variable for log-demand changes.
        theta (cp.Variable | None): Optional CVXPY Variable for policy parameters.
        tol (float): Relative tolerance for the NLP solver.

    Returns:
        PricingResult: Final profit, price changes, demand changes, and policy parameters.
    """
        
    obj = profit_data.r_nom @ cp.exp(delta + pi) - profit_data.kappa_nom @ cp.exp(delta)
    nlp = cp.Problem(cp.Maximize(obj), cvxpy_constraints)
                
    pi.value = np.zeros(n)
    delta.value = np.zeros(n)
    nlp.solve(solver=cp.IPOPT, nlp=True, verbose=False, derivative_test='none', tol=tol)
    
    return PricingResult(
        profit=_profit(profit_data, pi.value),
        price_changes=np.exp(pi.value), demand_changes=np.exp(delta.value),
        policy_parameters=theta.value if theta is not None else None
    )


def solve_ppp(
    profit_data: ProfitData, constraint_data: ConstraintData,
    method: str='QMM', tol: float=1e-3
) -> PricingResult:
    """Solve product pricing problem (PPP).
    
    Args:
        profit_data (ProfitData): nominal revenue, nominal cost, and elasticity matrix.
        constraint_data (ConstraintData): bounds and constraints.
        method (str): Optimization method to use ('QMM', 'CCP', or 'NLP'; defaults to 'QMM').
        tol (float): Relative improvement tolerance for termination, defaults to 1e-3.

    Returns:
        PricingResult: Final profit, price changes, demand changes, and policy parameters.
    """
    
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
        return _qmm(n, profit_data, constraint_data, constraints, pi, delta, theta, tol)
    elif method == 'CCP':
        return _ccp(n, profit_data, constraints, pi, delta, theta, tol)
    elif method == 'NLP':
        return _nlp(n, profit_data, constraints, pi, delta, theta, tol)
    else:
        raise ValueError(f"Unknown solve method: {method}")
