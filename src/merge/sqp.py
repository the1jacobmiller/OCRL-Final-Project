import osqp
import numpy as np


class SQPProblem:
    def __init__(
        self, P: np.ndarray, q: np.ndarray, l=np.ndarray, u=np.ndarray
    ) -> None:
        self.qp_problem = osqp.OSQP()

    def solve(self):
        return self.qp_problem.solve()

    def update(self, q_new, l_new, u_new):
        self.qp_problem.update(q=q_new, l=l_new, u=u_new)

    def sequential_step(self, P: np.ndarray, q: np.ndarray, l=np.ndarray, u=np.ndarray):
        pass
