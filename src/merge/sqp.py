import osqp
import numpy as np
from typing import Callable
from rich.progress import track


class SQPProblem:
    """
    Reference the SQP formulation from Lecture 12 here: https://github.com/Optimal-Control-16-745/lecture-notebooks/blob/main/Lecture%2012/Lecture%2012.pdf
    """
    def setup_qp_and_solve(
        self, P: np.ndarray, q: np.ndarray, A: np.ndarray, l: np.ndarray, u: np.ndarray
    ):
        self.qp_problem = osqp.OSQP()
        self.qp_problem.setup(P=P, q=q, A=A, l=l, u=u)
        return self.qp_problem.solve()

    def sequential_step(
        self,
        cost: np.ndarray,
        gradient_of_lagrangian: np.ndarray,
        hessian_of_lagrangian: np.ndarray,
        equality_constraint: np.ndarray,
        inequality_constraint: np.ndarray,
        gradient_equality: np.ndarray,
        gradient_inequality: np.ndarray,
    ):
        # TODO: create the qp from the above functions
        P = None
        q = None
        A = None
        l = None
        u = None

        result = self.setup_qp_and_solve(P, q, A, l, u)
        delta_z = result.x
        return delta_z

    def sequential_solve(
        self, initial_x, initial_u, initial_lambda, initial_mu, max_iters=100, tolerance=1e-2
    ):

        sqp_x = initial_x
        sqp_u = initial_u
        sqp_lambda = initial_lambda
        sqp_mu = initial_mu

        # backtracking parameter
        alpha = 1.0

        # outer loop for the quadratic approximation

        for x_idx in track(range(max_iters)):
            cost = lambda x: x
            gradient_of_cost = lambda x: x
            gradient_of_lagrangian = np.zeros(10)
            hessian_of_lagrangian = np.zeros(10)
            equality_constraint = np.zeros(10)
            inequality_constraint = np.zeros(10)
            gradient_equality = np.zeros(10)
            gradient_inequality = np.zeros(10)

            # solve for this timestep's search direction using OSQP
            delta_z = self.sequential_step(
                cost,
                gradient_of_lagrangian,
                hessian_of_lagrangian,
                equality_constraint,
                inequality_constraint,
                gradient_equality,
                gradient_inequality,
            )

            if np.linalg.norm(delta_z) < tolerance:
                break

            # TODO: fix this
            delta_x = delta_z[0]
            delta_u = delta_z[0]
            delta_lambda = delta_z[1]
            delta_mu = delta_z[2]

            # inner loop for the line search
            if cost(sqp_x + delta_x) > cost(
                sqp_x
            ) + tolerance * alpha * gradient_of_cost(sqp_x):
                alpha = 0.5 * alpha

            # update the x based on the line search
            sqp_x += alpha * delta_x
            sqp_u += alpha * delta_u
            # TODO: do we need to scale this multipliers by alpha as well?
            sqp_lambda += alpha * delta_lambda
            sqp_mu += alpha * delta_mu

        print(f"Solve took {x_idx} iterations")
        return sqp_x, sqp_u
