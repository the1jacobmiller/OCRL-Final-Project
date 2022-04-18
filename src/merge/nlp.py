from cProfile import label
import casadi as c
import matplotlib.pyplot as plt


class NLPProblem:
    """
    Uses IPOPT to solve the NLP we have. The obstacle constraints are the main reason we need to do this.
    """

    def __init__(
        self,
        x_init: list,
        x_goal: list,
        n: int = 4,
        m: int = 2,
        t_final:float=10,
        duration: int = 45,
        u_min: float = -1,
        u_max: float = 1,
    ) -> None:
        self.t_final = t_final
        self.duration = duration
        # the dynamics function
        self.dynamics = lambda x, u: c.vertcat(
            x[2],
            x[3],
            u[0],
            u[1]
        )

        # set up the problem in Casadi's Opti interface
        self.opti = c.Opti()

        # state and control
        x = self.opti.variable(n, duration)
        u = self.opti.variable(m, duration)

        # Cost matrices
        Q = c.diag(c.MX([1.2, 10, 2, 4]))
        Qf = c.MX.eye(4) * 100
        R = c.diag(c.MX([3, 3]))
        stage_cost = (x - x_goal).T @ Q @ (x - x_goal) + u.T @ R @ u
        term_cost = (x[:, -1] - x_goal).T @ Qf @ (x[:, -1] - x_goal)

        # const function
        self.opti.minimize(c.sumsqr(stage_cost) + term_cost)

        dt = t_final/duration
        for k in range(duration-1):
            # runga kutta forward simulation
            k1 = self.dynamics(x[:, k]           , u[:, k])
            k2 = self.dynamics(x[:, k] + dt/2*k1 , u[:, k])
            k3 = self.dynamics(x[:, k] + dt/2*k2 , u[:, k])
            k4 = self.dynamics(x[:, k] + dt*k3   , u[:, k])
            x_next = x[:, k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            # set the dynamics constraints
            self.opti.subject_to(x[:, k+1]==x_next)

        # set up the constraints
        self.opti.subject_to(x[:, 0] == x_init)
        self.opti.subject_to(self.opti.bounded(u_min, u, u_max))

        # create a view of the matrices
        self.x = x
        self.u = u

        self.opti.solver("ipopt")
        self.result = self.opti.solve()
    
    def get_state(self):
        return self.result.value(self.x)
    
    def get_controls(self):
        return self.result.value(self.u)
    
    def plot_solution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        state = self.get_state()
        control = self.get_controls()
        ax1.set_title("States")
        ax1.plot(state[0, :], 'r', label="X Position")
        ax1.plot(state[0, :], 'r--', label="X Velocity")
        ax1.plot(state[0, :], 'b', label="Y Position")
        ax1.plot(state[0, :], 'b--', label="Y Velocity")
        ax1.legend()
        # plot the controls
        ax2.set_title("Controls")
        ax2.plot(control[0, :], 'm', label="1st Control")
        ax2.plot(control[1, :], 'm', label="2nd Control")
        ax2.legend()


        plt.show()
        



if __name__ == "__main__":
    prob = NLPProblem([0, 0, 0, 0], [2, 3, 0.1, 1])
    prob.plot_solution()
