import cvxopt
import cvxpy as cp
import numpy as np
from pylab import axis, plot, show, title, xlabel, ylabel

from cvxpy import Maximize, Minimize, Problem, Variable


np.random.seed(42)


def chebysev_center(d, P, epsilon=1e-3):
    """Chebysev center

    Parameters:
        d: dimension
        P: set P
        epsilon: epsilon of set P

    Returns:
        radius: radius of the ball
        A: center
    """

    # Variables initialization
    radius = cp.Variable(1, pos=True)
    A = cp.Variable((d, d), PSD=True)
    l_P = len(P)
    constraints = []

    # Linear constraint
    if l_P > 0:
        for i, M in enumerate(P):
            constraints += [cp.sum((cp.multiply(M, A))) + radius * cp.sum((cp.multiply(M, M))) <= epsilon]

    # Bounded constraint
    constraints += [A << np.eye(d)]
    constraints += [radius <= 100]

    # Objective and solve
    objective = cp.Maximize(radius)
    p = cp.Problem(objective, constraints)
    result = p.solve()

    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return radius.value, A.value


def sdp_cost(x_t, x_0, P, epsilon=1e-3):
    # Variables initialization       
    A = cp.Variable((d, d), PSD=True)

    constraints = []

    # Linear constraint
    for i, M in enumerate(P):
        constraints += [cp.sum((cp.multiply(M, A))) <= epsilon]

    # Bounded constraint
    constraints += [A << np.eye(d)]

    # Objective and solve
    objective = cp.Maximize((x_t - x_0).T @ A @ (x_t - x_0))
    p = cp.Problem(objective, constraints)
    result = p.solve()

    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return A.value


if __name__ == '__main__':
    # variables
    d = 5
    # P = [np.random.rand(5, 5) for i in range(3)]
    P = []
    # P = np.array(P)
    alpha = 0.1

    radius, A_opt = chebysev_center(d, P, 0)
    print(radius, A_opt)

    x_t = np.random.rand(5)
    x_0 = np.random.rand(5)
    A_opt = sdp_cost(x_t, x_0, P)
    print(A_opt)
