from flow.controllers.base_controller import BaseController
from flow.envs import Env
from merge.nlp import NLPProblem
import numpy as np
import casadi as c
import copy
import os


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
        N=20,
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
        N: int, optional
            number of MPC time steps (default: 20)
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
        self.N = N
        self.t = 0
        self.plot_arr = []

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

    def get_accel(self, env: Env):
        vehicles = {}
        for veh_id in env.sorted_ids:
            edge = env.k.vehicle.get_edge(veh_id)
            edge_len = env.k.scenario.edge_length(edge)
            pos = env.k.vehicle.get_position(veh_id)
            speed = env.k.vehicle.get_speed(veh_id)
            vehicles[veh_id] = (edge, edge_len, pos, speed)

        min_seperation = env.k.vehicle.get_length(self.veh_id) * 2.0
        speed_limit = env.net_params.additional_params['speed_limit']
        target_speed = env.env_params.additional_params['target_velocity']
        accel_limit = env.env_params.additional_params['max_accel']

        x0, top_merge_indices, \
            bottom_merge_indices = self.get_observable_state(env, vehicles)
        Xref,Uref = self.get_reference_trajectory(env, x0, min_seperation,
                                                  target_speed, accel_limit,
                                                  bottom_merge_indices)

        controls = None
        nlp = NLPProblem()

        iter = 0
        while controls is None and iter < 3:
            try:
                x_res,u_res = nlp.solve(Xref,
                                        Uref,
                                        top_merge_indices,
                                        bottom_merge_indices,
                                        min_seperation,
                                        speed_limit,
                                        accel_limit,
                                        self.dt,
                                        self.N)
                controls = u_res[0][0]
            except:
                print('****************IPOPT SOLVE FAILED!!*******************')
                print('Iter %d: Relaxing constraints' % (iter))
                min_seperation = max(min_seperation*0.75, env.k.vehicle.get_length(self.veh_id))
                speed_limit = speed_limit + 5.
                accel_limit = accel_limit + 1.
            finally:
                iter += 1


        if controls is None:
            # We couldn't find a solution even after relaxing constraints - use
            # the reference controls
            print('****************USING REFERENCE CONTROLS!!*****************')
            controls = Uref[0][0]

        # Plotting code, saves tuple of controls and position to be loaded in by create_plots
        self.t += 1
        self.plot_arr.append((self.veh_id, self.t, controls, x0[0], x0[1]))
        np.save('merge/plot_data/' + self.veh_id + ".npy", self.plot_arr)

        print('Vehicle:', self.veh_id)
        print('Control:', controls)

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

        speed_limit = env.net_params.additional_params['speed_limit']
        dt = self.dt

        # Store the edge lengths
        inflow_highway_edge_len = env.k.scenario.edge_length('inflow_highway')
        inflow_merge_edge_len = env.k.scenario.edge_length('inflow_merge')
        left_edge_len = env.k.scenario.edge_length('left')
        bottom_edge_len = env.k.scenario.edge_length('bottom')
        center0_merge_edge_len = env.k.scenario.edge_length(':center_0')
        center1_merge_edge_len = env.k.scenario.edge_length(':center_1')
        bottom0_edge_len = env.k.scenario.edge_length(':bottom_0')
        left0_edge_len = env.k.scenario.edge_length(':left_0')

        edge_start_pos = {}
        edge_start_pos['inflow_highway'] = -(center1_merge_edge_len + \
                                             left_edge_len + \
                                             left0_edge_len + \
                                             inflow_highway_edge_len)
        edge_start_pos[':left_0'] = -(center1_merge_edge_len + \
                                      left_edge_len + \
                                      left0_edge_len)
        edge_start_pos['left'] = -(center1_merge_edge_len + left_edge_len)
        edge_start_pos[':center_1'] = -center1_merge_edge_len
        edge_start_pos['inflow_merge'] = -(center0_merge_edge_len + \
                                           bottom_edge_len + \
                                           bottom0_edge_len + \
                                           inflow_merge_edge_len)
        edge_start_pos[':bottom_0'] = -(center0_merge_edge_len + \
                                        bottom_edge_len + \
                                        bottom0_edge_len)
        edge_start_pos['bottom'] = -(center0_merge_edge_len + bottom_edge_len)
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

        if ego_edge in top_merge_edges:
            top_merge_indices.append(0)
        if ego_edge in bottom_merge_edges:
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

                # Don't add vehicles that are outside of the MPC time horizon.
                if abs(pos-ego_pos) > self.N*dt*speed_limit:
                    continue
                x0.extend([pos, value[SPEED]])

                vehicle_idx = len(x0)//2
                if edge in top_merge_edges:
                    top_merge_indices.append(vehicle_idx)
                if edge in bottom_merge_edges:
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

    @staticmethod
    def get_vehicle_separations(veh_idx, x):
        n = len(x)
        n_vehicles = n//2

        vehicle_separations = []
        for k in range(n_vehicles):
            if k != veh_idx:
                separation = x[2*veh_idx] - x[2*k]
                vehicle_separations.append(separation)

        return np.array(vehicle_separations)

    @staticmethod
    def clamp(min_num, num, max_num):
        return max(min(max_num, num), min_num)

    @staticmethod
    def get_reference_control(x, min_seperation, accel_limit,
                              target_speed, bottom_merge_indices):
        # Choose the controls for this time step. This approach puts the
        # responsibility to accelerate/decelerate on the merging vehicle,
        # keeping the reference velocity for highway vehicles constant.
        n = len(x)
        m = n//2
        uk = np.zeros((m,1))
        for i in range(m):
            if i in bottom_merge_indices and m > 1:
                vehicle_separations = NashController.get_vehicle_separations(i, x)
                min_seperation_idx = np.argmin(np.abs(vehicle_separations))

                # Check if vehicle i is too close to any other vehicles.
                if abs(vehicle_separations[min_seperation_idx]) < min_seperation:
                    if vehicle_separations[min_seperation_idx] > min_seperation/2.:
                        # Vehicle i is sufficiently ahead of the other
                        # vehicle - vehicle i should accelerate.
                        uk[i] = accel_limit
                    else:
                        # Vehicle i not sufficiently ahead or is behind the
                        # other vehicle - vehicle i should decelerate
                        uk[i] = -accel_limit
            if uk[i] == 0 and x[2*i+1] != target_speed:
                # Vehicle i is not at target speed - accelerate to target speed.
                speed_diff = target_speed - x[2*i+1]
                uk[i] = NashController.clamp(-accel_limit,
                                             speed_diff/10.,
                                             accel_limit)
        return uk

    def get_reference_trajectory(self, env, x0, min_seperation, target_speed,
                                 accel_limit, bottom_merge_indices):
        dt = self.dt
        n = len(x0)
        m = n//2

        A,B = NashController.get_dynamics_jacobians(m, dt)

        Xref = [np.array(x0).reshape((n,1))]
        Uref = []
        for k in range(1,self.N):
            uk = NashController.get_reference_control(Xref[-1],
                                                      min_seperation,
                                                      accel_limit,
                                                      target_speed,
                                                      bottom_merge_indices)
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
