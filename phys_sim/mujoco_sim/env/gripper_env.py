import os

import numpy as np

from mujoco_sim.env.base_env import RobotEnv


class FrankaEnv(RobotEnv):
    def __init__(
        self,
        xml_file=None,
        **kwargs,
    ):

        if xml_file is None:
            from clean_disentangle.utils.io_utils import PROJ_DIR

            xml_file = os.path.join(PROJ_DIR, "assets/mujoco/franka/scene.xml")

        super(FrankaEnv, self).__init__(
            xml_file=xml_file,
            **kwargs,
        )

        self.ctrl_mode = "relative"
        self.cam_views = [
            "view_front1",
            "view_front2",
            "view_back1",
            "view_back2",
            "top",
        ]

    def set_ctrl_mode(self, mode):
        self.ctrl_mode = mode

    def _set_action(self, action):
        assert action.shape == (8,)

        ctrl_range = self.mj_model.actuator_ctrlrange
        actuation_range = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0

        if self.ctrl_mode == "relative":
            actuation_center = np.zeros_like(action)
            for i in range(self.data.ctrl.shape[0]):
                actuation_center[i] = self.data.actuator(f"actuator{(i + 1)}").length[0]
                if i == 7:
                    actuation_center[i] = (actuation_center[i] / 0.4) * 255

            self.data.ctrl[:] = actuation_center + actuation_range * action
        else:
            self.data.ctrl[:] += actuation_range * action

        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrl_range[:, 0], ctrl_range[:, 1])


class UR5Env(RobotEnv):
    def __init__(
        self,
        xml_file=None,
        **kwargs,
    ):
        if xml_file is None:
            from clean_disentangle.utils.io_utils import PROJ_DIR

            xml_file = os.path.join(PROJ_DIR, "assets/mujoco/ur5e/scene.xml")

        super().__init__(
            xml_file=xml_file,
            **kwargs,
        )

        self.ctrl_mode = "relative"
        self.cam_views = [
            "view_front1",
            "view_front2",
            "view_back1",
            "view_back2",
            "top",
        ]

    def set_ctrl_mode(self, mode):
        self.ctrl_mode = mode

    def _set_action(self, action):
        raise NotImplementedError
        # assert action.shape == (6,)
        #
        # ctrl_range = self.model.actuator_ctrlrange
        # actuation_range = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.
        #
        # if self.ctrl_mode == "relative":
        #     actuation_center = np.zeros_like(action)
        #     for i in range(self.data.ctrl.shape[0]):
        #         actuation_center[i] = self.data.actuator(f'actuator{(i + 1)}').length[0]
        #         if i == 7:
        #             actuation_center[i] = (actuation_center[i] / 0.4) * 255
        #
        #     self.data.ctrl[:] = actuation_center + actuation_range * action
        # else:
        #     self.data.ctrl[:] += actuation_range * action
        #
        # self.data.ctrl[:] = np.clip(self.data.ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
