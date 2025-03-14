import os
import numpy as np
from .base_env import RobotEnv


class TrifingerEnv(RobotEnv):
    def __init__(
        self,
        xml_file=None,
        **kwargs,
    ):
        if xml_file is None:
            from clean_disentangle.utils.io_utils import PROJ_DIR

            xml_file = os.path.join(PROJ_DIR, "assets/mujoco/trifingerv2/trifinger.xml")

        super().__init__(xml_file=xml_file, **kwargs)

        self.ctrl_mode = "relative"
        self.cam_views = ["closeup"]

    def sample_rand_qpos(self):
        ctrl_range = self.mj_model.actuator_ctrlrange
        low_range = ctrl_range[:, 0]
        delta_range = ctrl_range[:, 1] - ctrl_range[:, 0]

        n_joints = len(ctrl_range)
        weights = np.random.random((n_joints,))
        rand_qpos = low_range + weights * delta_range

        return rand_qpos

    def set_ctrl_mode(self, mode):
        self.ctrl_mode = mode


class DclawEnv(RobotEnv):
    def __init__(self, xml_file=None, **kwargs):
        if xml_file is None:
            from clean_disentangle.utils.io_utils import PROJ_DIR

            xml_file = os.path.join(PROJ_DIR, "assets/mujoco/dclaw/scene.xml")

        super().__init__(xml_file=xml_file, **kwargs)

        self.ctrl_mode = "relative"
        self.cam_views = ["closeup"]

    def set_ctrl_mode(self, mode):
        self.ctrl_mode = mode
