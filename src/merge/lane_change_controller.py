from flow.controllers.base_lane_changing_controller import BaseLaneChangeController
from flow.envs import Env


class LaneChangeController(BaseLaneChangeController):
    """A lane-changing model used to move vehicles into lane 0."""

    def get_lane_change_action(self, env: Env):
        """
        Can return [-1, 0, 1] in order to change lanes
        """
        current_lane = env.k.vehicle.get_lane(self.veh_id)
        if current_lane > 0:
            return -1
        else:
            return 0
