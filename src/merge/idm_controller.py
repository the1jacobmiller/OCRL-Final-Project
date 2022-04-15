from flow.controllers.base_controller import BaseController
import numpy as np


class IDMController(BaseController):
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

    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # negative headways may be registered by sumo at intersections/
        # junctions. Setting them to 0 causes vehicles to not move; therefore,
        # we maintain these negative headways to let sumo control the dynamics
        # as it sees fit at these points.
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == "":  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) / (2 * np.sqrt(self.a * self.b))
            )

        return self.a * (1 - (v/self.v0)**self.delta - (s_star/h)**2)
