import casadi as c
import numpy as np
import matplotlib.pyplot as plt


class NLPProblem:
    """
    Uses IPOPT to solve the NLP we have. The obstacle constraints are the main reason we need to do this.
    """

    def solve(self,
              Xref,
              Uref,
              top_merge_indices,
              bottom_merge_indices,
              min_seperation,
              speed_limit,
              accel_limit,
              dt,
              N):
        self.opti = c.Opti()
        self.dt = dt
        self.N = N

        n = Xref.shape[1]
        m = Uref.shape[1]
        n_vehicles = n//2

        self.x = self.opti.variable(n,self.N)
        self.u = self.opti.variable(m,self.N)

        Q = c.MX.eye(n)
        Qf = c.MX.eye(n)
        R = c.MX.eye(self.N-1)

        # Allow vehicles on the bottom merge lanes to use more aggressive
        # control.
        if 0 in bottom_merge_indices:
            R = R * 1e-3

        # Allow vehicles on the bottom merge lanes to deviate more from the
        # reference trajectory.
        for k in range(n_vehicles):
            if k in bottom_merge_indices:
                Q[2*k,2*k] = 1e-2 # position
                Q[2*k+1,2*k+1] = 1e-1 # velocity
                Qf[2*k,2*k] = 1e-2 # final position
                Qf[2*k+1,2*k+1] = 1e-1 # final velocity
            else:
                Q[2*k,2*k] = 1e3 # position
                Q[2*k+1,2*k+1] = 1e5 # velocity
                Qf[2*k,2*k] = 1e3 # final position
                Qf[2*k+1,2*k+1] = 1e5 # final velocity


        Xref = Xref.T.squeeze(0)
        Uref = Uref.T.squeeze(0)

        stage_cost = (self.x - Xref).T @ Q @ (self.x - Xref) + (self.u[0] - Uref[0,:]).T @ R @ (self.u[0] - Uref[0,:])
        term_cost = (self.x[:,-1] - Xref[:,-1]).T @ Qf @ (self.x[:,-1] - Xref[:,-1])

        # const function
        self.opti.minimize(c.sumsqr(stage_cost) + term_cost)

        # set the acceleration constraints
        self.opti.subject_to(self.opti.bounded(-accel_limit, self.u, accel_limit))

        A, B = NLPProblem.get_dynamics_jacobians(self.dt, n_vehicles)

        for k in range(self.N-1):
            # set the dynamics constraints
            self.opti.subject_to(self.x[:,k+1]==A@self.x[:,k] + B@self.u[:,k])

        for k in range(n_vehicles):
            # set the velocity constraints
            self.opti.subject_to(self.opti.bounded(0, self.x[2*k+1,:], speed_limit))

        for i in range(0, n_vehicles-1):
            xi = self.x[2*i,:]
            for j in range(i+1, n_vehicles):
                xj = self.x[2*j,:]
                if (i in top_merge_indices and j in bottom_merge_indices) \
                    or (j in top_merge_indices and i in bottom_merge_indices):
                    # Calculate the start separation, and give a little wiggle
                    # room to avoid infeasiblity at the start.
                    start_separation = (Xref[0][2*i] - Xref[0][2*j])**2 - 1.0
                    if (start_separation > min_seperation**2):
                        # Treat it like the normal case - constraint is fully
                        # active at all times.
                        constraint = (xj - xi)**2
                        self.opti.subject_to(self.opti.bounded(min_seperation**2,
                                                               constraint,
                                                               np.inf))
                    else:
                        time_to_merge = min(-Xref[0][2*i], -Xref[0][2*j])/speed_limit
                        time_steps_to_merge = min(int(time_to_merge/self.dt), self.N)

                        alpha = (min_seperation**2 - start_separation)/time_steps_to_merge
                        tau = start_separation
                        for t in range(self.N):
                            # ramp up tau to avoid infeasibility
                            if tau < min_seperation**2:
                                tau += alpha
                            else:
                                tau = min_seperation**2
                            constraint = (xi[t] - xj[t])**2
                            self.opti.subject_to(self.opti.bounded(tau, constraint, np.inf))
                else:
                    # needs to be tuned for normal conditions
                    tau = min_seperation**2
                    constraint = (xj - xi)**2
                    self.opti.subject_to(self.opti.bounded(tau, constraint, np.inf))

        self.opti.solver("ipopt")
        self.result = self.opti.solve()

        x_res = self.get_state()
        u_res = self.get_controls()

        return x_res, u_res

    @staticmethod
    def get_dynamics_jacobians(dt, n_vehicles):

        a = c.DM([[1, dt],[0, 1]])
        A = a

        for _ in range(1, n_vehicles):
            A = c.diagcat(A, a)

        b = c.DM([[0],[dt]])
        B = b

        for _ in range(1, n_vehicles):
            B = c.diagcat(B, b)

        return A,B

    def get_state(self):
        return self.result.value(self.x).reshape(self.x.shape)

    def get_controls(self):
        return self.result.value(self.u).reshape(self.u.shape)

    def plot_solution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        state = self.get_state()
        control = self.get_controls()
        ax1.set_title("States")
        ax1.plot(state[0, :], 'r', label="X Position")
        ax1.plot(state[1, :], 'r--', label="X Velocity")
        ax1.plot(state[2, :], 'b', label="Y Position")
        ax1.plot(state[3, :], 'b--', label="Y Velocity")
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
