import cvxopt
import cvxpy as cp
import numpy as np
from pylab import axis, plot, show, title, xlabel, ylabel

from cvxpy import Maximize, Minimize, Problem, Variable
from cvxpy.atoms.affine.upper_tri import vec_to_upper_tri


np.random.seed(42)


def chebysev_center(d, P, epsilon=1e-3, init=False):
    """Chebysev center

    Parameters:
        d: dimension
        P: set P
        epsilon: epsilon of set P

    Returns:
        radius: radius of the ball
        A: Chebysev center
    """

    # Variables initialization
    radius = cp.Variable(1, pos=True)
    A = cp.Variable((d, d), PSD=True)
    l_P = len(P)
    constraints = []

    # Linear constraint
    if l_P > 0:
        for i, M in enumerate(P):
            constraints += [cp.sum((cp.multiply(A, M))) + radius * cp.norm(M, 'fro') <= epsilon]

    # Bounded constraint
    constraints += [A << np.eye(d)]
    if init:
        constraints += [radius <= 1]

    # Objective and solve
    objective = cp.Maximize(radius)
    p = cp.Problem(objective, constraints)
    result = p.solve()

    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return radius.value, A.value


def sdp_cost(x_t, x_0, P, epsilon=1e-3):
    # Variables initialization       
    d = x_0.shape[0]
    A = cp.Variable((d, d), PSD=True)

    constraints = []

    # Linear constraint
    for i, M in enumerate(P):
        constraints += [cp.sum((cp.multiply(A, M))) <= epsilon]

    # Bounded constraint
    constraints += [A << np.eye(d)]

    # Objective and solve
    objective = cp.Maximize((x_t - x_0).T @ A @ (x_t - x_0))
    p = cp.Problem(objective, constraints)
    result = p.solve()

    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return A.value


def enclosing_heuristics(A_opt, P, epsilon):
    # Shape
    d = A_opt.shape[0]
    upper_d = d * (d + 1) // 2

    # Variables
    V = cp.Variable((d, d), symmetric=True)
    v = cp.Variable((upper_d, 1))
    Lmbda = cp.Variable((upper_d, upper_d), PSD=True)

    # Constraints
    constraints = []

    # \inner{A_c\opt + V}{M_{ij}} \le \eps ~\forall(i,j) \in \PP
    constraints += [(cp.sum((cp.multiply(A_opt + V, M))) <= epsilon) for M in P
]

    # 0 \preceq A_c\opt + V \preceq I
    constraints += [(A_opt + V) >> 0]
    constraints += [(A_opt + V) << np.eye(d)]
    # constraints += [Lmbda << 100 * np.eye(upper_d)]
    # constraints += [cp.trace(Lmbda) <= 1000000]

    # v = \mathrm{upper}(2V - \diag(V))
    constraints += [v == cp.upper_tri(2 * V - cp.multiply(V, np.eye(d)))]

    # [[Lmbda v] [v^T 1] >> 0
    constraints += [cp.bmat([[Lmbda, v], [cp.transpose(v), cp.reshape(1, (1, 1))]]) >> 0]

    # Objective and solve 
    objective = cp.Maximize(cp.trace(Lmbda))
    p = cp.Problem(objective, constraints)
    result = p.solve(solver=cp.MOSEK)

    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return V.value 


if __name__ == '__main__':
    # variables
    d = 5
    # P = [np.random.rand(5, 5) for i in range(3)]
    P = []
    # P = np.array(P)
    alpha = 0.1

    # radius, A_opt = chebysev_center(d, P, 0)
    # print(radius, A_opt)

    # x_t = np.random.rand(5)
    # x_0 = np.random.rand(5)
    # A_opt = sdp_cost(x_t, x_0, P)
    # print(A_opt)

    A_opt = np.random.rand(d, d)
    A_opt = A_opt @ A_opt.T
    A_opt = A_opt / max(np.linalg.eig(A_opt)[0])

    P = [np.random.rand(d, d) for i in range(2)]
    V = enclosing_heuristics(A_opt, P, 1e-6)
    print(np.linalg.eig(A_opt + V)[0], np.linalg.eig(np.eye(d) - A_opt - V)[0])
