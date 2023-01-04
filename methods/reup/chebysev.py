import cvxopt
import cvxpy as cp
import numpy as np
from pylab import axis, plot, show, title, xlabel, ylabel

from cvxpy import Maximize, Minimize, Problem, Variable


def chebysev_center(d, P, alpha=None):
    """Chebysev center

    Parameters:
        d: dimension
        P: set P
        alpha: inconsistent parameter

    Returns:
        radius: radius of the ball
        A: center
    """

    # Variables initialization
    radius = Variable(1)
    A = cp.Variable((d, d), PSD=True)
    l_P = len(P)
    constraints = []

    gamma_P = []
    for i in range(len(P)):
        gamma_P.append(cp.Variable(1))

    # Linear constraint
    if l_P > 0:
        for i, M in enumerate(P):
            constraints += [cp.sum((cp.multiply(M, A))) + radius * cp.sum((cp.multiply(M, M))) <= gamma_P[i]]

    # Inconsistent
    if l_P > 0:
        constraints += [cp.sum(gamma_P) <= alpha * len(P)]

    # Bounded constraint
    constraints += [radius >= 1e-9]
    constraints += [A << np.eye(d)]

    # Objective and solve
    objective = cp.Minimize(radius)
    p = cp.Problem(objective, constraints)
    result = p.solve()

    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return radius.value, A.value
    

if __name__ == '__main__':
    # variables
    d = 5
    P = np.random.rand(5, 5)
    P = np.expand_dims(P, axis=0)
    alpha = 0.1

    radius, A_opt = chebysev_center(d, P, 0.1)
    print(radius, A_opt)
