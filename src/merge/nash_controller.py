from flow.controllers.base_controller import BaseController
from flow.envs import Env
from merge.nlp import NLPProblem
import numpy as np
import casadi as c


class NashController(BaseController):

    def __init__(
        self,
        veh_id,
        v0=30,
        T=1,
        a=1,
        b=1.5,
        delta=4,
        s0=2,
        s1=0,
        time_delay=0.0,
        dt=0.1,
        noise=0,
        fail_safe=None,
        car_following_params=None,
    ):
        """
        veh_id: str
            unique vehicle identifier
        car_following_params: SumoCarFollowingParams
            see parent class
        v0: float, optional
            desirable velocity, in m/s (default: 30)
        T: float, optional
            safe time headway, in s (default: 1)
        b: float, optional
            comfortable deceleration, in m/s2 (default: 1.5)
        delta: float, optional
            acceleration exponent (default: 4)
        s0: float, optional
            linear jam distance, in m (default: 2)
        s1: float, optional
            nonlinear jam distance, in m (default: 0)
        dt: float, optional
            timestep, in s (default: 0.1)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        fail_safe: str, optional
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        """
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.s1 = s1
        self.dt = dt

        # Define the observable edges
        self.observable_edges = {}
        self.observable_edges['inflow_highway'] =   ['inflow_highway',
                                                    ':left_0',
                                                    'left']

        self.observable_edges[':left_0'] =          ['inflow_highway',
                                                    ':left_0',
                                                    'left']

        self.observable_edges['left'] =             ['inflow_highway',
                                                    ':left_0',
                                                    'left',
                                                    ':center_1',
                                                    'bottom',
                                                    ':center_0']

        self.observable_edges[':center_1'] =        ['left',
                                                    ':center_1',
                                                    'bottom',
                                                    ':center_0',
                                                    'center']

        self.observable_edges['inflow_merge'] =     ['inflow_merge',
                                                    ':bottom_0',
                                                    'bottom']

        self.observable_edges[':bottom_0'] =        ['inflow_merge',
                                                    ':bottom_0',
                                                    'bottom']

        self.observable_edges['bottom'] =           ['inflow_merge',
                                                    ':bottom_0',
                                                    'bottom',
                                                    ':center_0',
                                                    'left',
                                                    ':center_1']

        self.observable_edges[':center_0'] =        ['bottom',
                                                    ':center_0',
                                                    'left',
                                                    ':center_1',
                                                    'center']

        self.observable_edges['center'] =           [':center_0',
                                                    ':center_1',
                                                    'center']

    @staticmethod
    def getAB(dt, n_vehicles):

        a = c.DM([[1, dt],[0, 1]])
        A = a

        for _ in range(1, n_vehicles):
            A = c.diagcat(A, a)

        b = c.DM([[0],[dt]])
        B = b

        for _ in range(1, n_vehicles):
            B = c.diagcat(B, b)

        return A,B

    def get_accel(self, env: Env):
        vehicles = {}
        for veh_id in env.sorted_ids:
            edge = env.k.vehicle.get_edge(veh_id)
            edge_len = env.k.scenario.edge_length(edge)
            pos = env.k.vehicle.get_position(veh_id)
            speed = env.k.vehicle.get_speed(veh_id)
            vehicles[veh_id] = (edge, edge_len, pos, speed)

        #print(vehicles)

        x0, top_merge_indices, \
            bottom_merge_indices = self.get_observable_state(env, vehicles)
        Xref,Uref = self.get_reference_trajectory(env, x0)

        # TODO: perform the optimization problem here and return the acceleration control
        self.opti = c.Opti()

        n = Xref.shape[1] 
        m = Uref.shape[1]

        n_vehicles = n//2

        T = Xref.shape[0]

        x = self.opti.variable(n,T) 
        u = self.opti.variable(m,T)

        Q = c.MX.eye(n) * 100
        Qf = c.MX.eye(n) * 100
        R = c.MX.eye(m) * 100

        Xref = Xref.T.squeeze(0)
        Uref = Uref.T.squeeze(0)

        stage_cost = (x - Xref).T @ Q @ (x - Xref) + (u - Uref).T @ R @ (u - Uref)
        term_cost = (x[:,-1] - Xref[:,-1]).T @ Qf @ (x[:,-1] - Xref[:,-1])
    
        # const function
        self.opti.minimize(c.sumsqr(stage_cost) + term_cost)

        speed_limit = env.net_params.additional_params['speed_limit']
        accel_limit = env.env_params.additional_params['max_accel']

        # set the acceleration constraints
        self.opti.subject_to(self.opti.bounded(-accel_limit, u, accel_limit))

        A, B = NashController.getAB(self.dt, n_vehicles)

        for k in range(T-1):
            # set the dynamics constraints
            self.opti.subject_to(x[:,k+1]==A@x[:,k] + B@u[:,k])

        for k in range(T):
            # set the velocity constraints
            self.opti.subject_to(self.opti.bounded(0, x[:,k][1], speed_limit))


        # I had to iterate over a list of the keys because the 
        # id name is flow_00_x for cars on the inflow and 
        # flow_10_x for cars on the merge making it hard to index
        # also doing it this way because top/bottom_merge_indices
        # weren't returning anything 
        veh_keys = list(vehicles.keys())
        
        for i in range(0, n_vehicles-1):
            xi = x[i,:]
            edge_i = env.k.vehicle.get_edge(veh_keys[i])
            for j in range(i+1, n_vehicles):
                xj = x[j,:]
                edge_j = env.k.vehicle.get_edge(veh_keys[j])
                if edge_i == 'bottom' and edge_j == 'left' or edge_i == 'left' and edge_j == 'bottom':
                    alpha = 0.9
                    tau = 10
                    for t in range(T):
                        # ramp down tau to avoid infeasible conditions
                        constraint = np.sqrt((xi[t] - xj[t])**2)
                        tau *= alpha
                        print("tau: ", tau)
                        self.opti.subject_to(self.opti.bounded(tau, constraint, np.inf))
                else:
                    # needs to be tuned for normal conditions
                    tau = 1.0
                    constraint = (xj - xi)**2
                    self.opti.subject_to(self.opti.bounded(tau, constraint, np.inf))

        self.opti.solver("ipopt")
        self.result = self.opti.solve()

        x_res = self.result.value(x)
        u_res = self.result.value(u)

        x_res = x_res.reshape(x.shape)
        u_res = u_res.reshape(u.shape)

        # print(x_res[0,:])
        print(u_res[0,:])

        controls = u_res[0][0]

        #controls = 0
        return controls

    def get_observable_state(self, env, vehicles):
        # Notes:
        # - What other edges vehicles can observe can be configured in
        # 'observable_edges'.
        # - This assumes we can reach a reasonable equilibrium when each
        # vehicle only reasons about other vehicles on nearby relevant lanes
        # rather than having knowledge of every vehicle.
        # - Vehicles in a merge lane should reason about all other vehicles
        # in the connected merge lanes. However, the collision constraints
        # should not go into effect immediately when the vehicle enters the
        # merge lane, as this could result in an infeasible problem if two
        # vehicles entered adjacent merge lanes at the same time.
        # - x=0 at the merge. The position of any vehicles before the merge is
        # negative.
        # - Because of how FLOW defines lanes, this will only work with a merge
        # scenario that has one highway lane and one merge lane.
        EDGE = 0
        EDGE_LEN = 1
        POSITION = 2
        SPEED = 3

        # Store the edge lengths
        inflow_highway_edge_len = env.k.scenario.edge_length('inflow_highway')
        inflow_merge_edge_len = env.k.scenario.edge_length('inflow_merge')
        merge_edge_len = env.k.scenario.edge_length('left')
        center0_merge_edge_len = env.k.scenario.edge_length(':center_0')
        center1_merge_edge_len = env.k.scenario.edge_length(':center_1')
        bottom0_edge_len = env.k.scenario.edge_length(':bottom_0')
        left0_edge_len = env.k.scenario.edge_length(':left_0')

        assert merge_edge_len == env.k.scenario.edge_length("bottom")

        edge_start_pos = {}
        edge_start_pos['inflow_highway'] = -(center1_merge_edge_len + \
                                             merge_edge_len + \
                                             left0_edge_len + \
                                             inflow_highway_edge_len)
        edge_start_pos[':left_0'] = -(center1_merge_edge_len + \
                                      merge_edge_len + \
                                      left0_edge_len)
        edge_start_pos['left'] = -(center1_merge_edge_len + merge_edge_len)
        edge_start_pos[':center_1'] = -center1_merge_edge_len
        edge_start_pos['inflow_merge'] = -(center0_merge_edge_len + \
                                           merge_edge_len + \
                                           bottom0_edge_len + \
                                           inflow_merge_edge_len)
        edge_start_pos[':bottom_0'] = -(center0_merge_edge_len + \
                                        merge_edge_len + \
                                        bottom0_edge_len)
        edge_start_pos['bottom'] = -(center0_merge_edge_len + merge_edge_len)
        edge_start_pos[':center_0'] = -center0_merge_edge_len
        edge_start_pos['center'] = 0

        top_merge_edges = ['left', ':center_1']
        bottom_merge_edges = ['bottom', ':center_0']

        x0 = None
        top_merge_indices = [] # indices of vehicles on the top merge lanes
        bottom_merge_indices = [] # indices of vehicles on the bottom merge lanes

        # First handle the ego vehicle
        ego_edge = vehicles[self.veh_id][EDGE]
        ego_pos = vehicles[self.veh_id][POSITION] + edge_start_pos[ego_edge]
        x0 = [ego_pos, vehicles[self.veh_id][SPEED]]

        if ego_edge in top_merge_indices:
            top_merge_indices.append(0)
        if ego_edge in bottom_merge_indices:
            bottom_merge_indices.append(0)

        # Now handle all other vehicles
        for key, value in vehicles.items():
            if key == self.veh_id:
                continue

            edge = value[EDGE]
            if edge in self.observable_edges[ego_edge]:
                # This vehicle is on an observable edge - add its
                # position and velocity to x0.
                pos = value[POSITION] + edge_start_pos[edge]
                x0.extend([pos, value[SPEED]])

                vehicle_idx = len(x0)//2
                if edge in top_merge_indices:
                    top_merge_indices.append(vehicle_idx)
                if edge in bottom_merge_indices:
                    bottom_merge_indices.append(vehicle_idx)

        return x0, top_merge_indices, bottom_merge_indices

    @staticmethod
    def get_dynamics_jacobians(m, dt):
        Ak = np.array([[1., dt],
                       [0., 1.]])
        Bk = np.array([[0.],
                       [dt]])
        A = Ak
        B = Bk
        for k in range(1,m):
            A = np.block([[A, np.zeros((A.shape[0],2))],
                          [np.zeros((2,A.shape[1])), Ak]])
            B = np.block([[B, np.zeros((B.shape[0],1))],
                          [np.zeros((2,B.shape[1])), Bk]])
        return A,B

    def get_reference_trajectory(self, env, x0):
        dt = 0.1
        N = 20
        n = len(x0)
        m = n//2

        A,B = NashController.get_dynamics_jacobians(m, dt)

        Xref = [np.array(x0).reshape((n,1))]
        Uref = [np.zeros((m,1))]
        for k in range(1,N):
            # Choose the controls for this time step
            # TODO: choose a control to meet desired action (e.g. avoid
            # collisions, accel to target speed)
            uk = np.zeros((m,1))
            Uref.append(uk)

            # Apply dynamics
            xk = A @ Xref[-1] + B @ Uref[-1]
            Xref.append(xk)

        Xref = np.array(Xref)
        Uref = np.array(Uref)

        return Xref, Uref

    def get_action(self, env):
        """Convert the get_accel() acceleration into an action.
        If no acceleration is specified, the action returns a None as well,
        signifying that sumo should control the accelerations for the current
        time step.
        This method also augments the controller with the desired level of
        stochastic noise, and utlizes the "instantaneous" or "safe_velocity"
        failsafes if requested.
        Parameters
        ----------
        env : flow.envs.Env
            state of the environment at the current time step
        Returns
        -------
        float
            the modified form of the acceleration
        """
        # this is to avoid abrupt decelerations when a vehicle has just entered
        # a network and it's data is still not subscribed
        if len(env.k.vehicle.get_edge(self.veh_id)) == 0:
            return None

        # Note: this is commented out so that we can control the vehicles in
        # the merge junction
        # this allows the acceleration behavior of vehicles in a junction be
        # described by sumo instead of an explicit model
        # if env.k.vehicle.get_edge(self.veh_id)[0] == ":":
        #   return None

        accel = self.get_accel(env)

        # if no acceleration is specified, let sumo take over for the current
        # time step
        if accel is None:
            return None

        # add noise to the accelerations, if requested
        if self.accel_noise > 0:
            accel += np.random.normal(0, self.accel_noise)

        # run the failsafes, if requested
        if self.fail_safe == 'instantaneous':
            accel = self.get_safe_action_instantaneous(env, accel)
        elif self.fail_safe == 'safe_velocity':
            accel = self.get_safe_velocity_action(env, accel)

        return accel
