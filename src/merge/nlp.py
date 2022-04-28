import casadi as c
import numpy as np
import matplotlib.pyplot as plt


class NLPProblem:
    """
    Uses IPOPT to solve the NLP we have. The obstacle constraints are the main reason we need to do this.
    """

    def solve(self,
              x0,
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

        n = Xref.shape[0]
        m = Uref.shape[0]
        n_vehicles = n//2

        self.x = self.opti.variable(n,self.N)
        self.u = self.opti.variable(m,self.N)

        Q = c.MX.eye(n)
        Qf = c.MX.eye(n)
        R = c.MX.eye(self.N)

        # Allow vehicles on the bottom merge lanes to use more aggressive
        # control.
        if 0 in bottom_merge_indices:
            R = R * 1e-3

        # Allow vehicles on the bottom merge lanes to deviate more from the
        # reference trajectory.
        for k in range(n_vehicles):
            if k in bottom_merge_indices:
                Q[2*k,2*k] = 1e-2 # position
                Q[2*k+1,2*k+1] = 1e1 # velocity
                Qf[2*k,2*k] = 1e-2 # final position
                Qf[2*k+1,2*k+1] = 1e1 # final velocity
            else:
                Q[2*k,2*k] = 1e3 # position
                Q[2*k+1,2*k+1] = 1e5 # velocity
                Qf[2*k,2*k] = 1e3 # final position
                Qf[2*k+1,2*k+1] = 1e5 # final velocity

        stage_cost = (self.x - Xref).T @ Q @ (self.x - Xref) + self.u[0,:] @ R @ self.u[0,:].T
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
                    start_separation = (x0[2*i] - x0[2*j])**2 - 1.0
                    if (start_separation > min_seperation**2):
                        # Treat it like the normal case - constraint is fully
                        # active at all times.
                        constraint = (xj - xi)**2
                        self.opti.subject_to(self.opti.bounded(min_seperation**2,
                                                               constraint,
                                                               np.inf))
                    else:
                        time_to_merge = min(-x0[2*i], -x0[2*j])/speed_limit
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

        self.opti.set_initial(self.x, Xref)
        self.opti.set_initial(self.u, Uref)
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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15,15))
        state = self.get_state()
        control = self.get_controls()
        n_vehicles = state.shape[0]//2

        ax1.set_title("Positions")
        for i in range(n_vehicles):
            ax1.plot(state[2*i, :], label="Veh #"+str(i))
        ax1.ticklabel_format(useOffset=False, style='plain')
        ax1.legend()

        ax2.set_title("Velocities")
        for i in range(n_vehicles):
            ax2.plot(state[2*i+1, :], label="Veh #"+str(i))
        ax2.ticklabel_format(useOffset=False, style='plain')
        ax2.legend()

        ax3.set_title("Accelerations")
        for i in range(n_vehicles):
            ax3.plot(control[i, :], label="Veh #"+str(i))
        ax3.legend()
        plt.show()

    @staticmethod
    def plot_reference_trajectories(Xref, Uref):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15,15))
        n_vehicles = Xref.shape[0]//2

        ax1.set_title("Reference Positions")
        for i in range(n_vehicles):
            ax1.plot(Xref[2*i, :], label="Veh #"+str(i))
        ax1.ticklabel_format(useOffset=False, style='plain')
        ax1.legend()

        ax2.set_title("Reference Velocities")
        for i in range(n_vehicles):
            ax2.plot(Xref[2*i+1, :], label="Veh #"+str(i))
        ax2.ticklabel_format(useOffset=False, style='plain')
        ax2.legend()

        ax3.set_title("Reference Accelerations")
        for i in range(n_vehicles):
            ax3.plot(Uref[i, :], label="Veh #"+str(i))
        ax3.legend()
        plt.show()
