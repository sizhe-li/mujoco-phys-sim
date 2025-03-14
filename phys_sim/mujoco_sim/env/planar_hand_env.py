import copy
import random
from dataclasses import dataclass, field
from typing import List, Literal, Tuple
from xml.etree import ElementTree

import mujoco
import numpy as np
from mujoco_sim.env.base_env import RobotEnv, RobotEnvCfg
from mujoco_sim.models import MujocoWorldBase
from mujoco_sim.models.arenas.empty_arena import EmptyArena
from mujoco_sim.models.objects import MeshObject
from robosuite.controllers import JointPositionController
from robosuite.controllers.interpolators.linear_interpolator import LinearInterpolator
from robosuite.models.objects import BoxObject, CylinderObject, MujocoObject
from robosuite.utils.mjcf_utils import CustomMaterial, new_actuator
from robosuite.utils.placement_samplers import UniformRandomSampler


@dataclass
class PlanarHandEnvCfg(RobotEnvCfg):
    # builder_cfg: PushEnvBuilderCfg
    # sampler_cfg: PushEnvObjectSamplerCfg
    pass


class PlanarHandEnv(RobotEnv):
    cfg: PlanarHandEnvCfg
    mj_objects: List[MujocoObject]
    object_sampler: UniformRandomSampler | None

    def __init__(
        self,
        cfg: PlanarHandEnvCfg,
    ):
        self.cfg = cfg

        super().__init__(cfg=cfg, mj_model=None)

        self.build_controller()

    def build_controller(self):

        self.interpolator = LinearInterpolator(
            ndim=2, controller_freq=32, policy_freq=1, ramp_ratio=1.0
        )

        self.controller = JointPositionController(
            self.sim,
            eef_name="pusher_default_site",
            joint_indexes={
                "joints": [0, 1],
                "qpos": [0, 1],
                "qvel": [0, 1],
            },
            actuator_range=(-9999, -9999),  # not used, placeholders
            output_max=1.0,
            output_min=-1.0,
            interpolator=self.interpolator,
        )

    def _pre_action(self, action, policy_step=False):

        if policy_step:
            # force absolute position-based control
            # the first part is merely a placeholder
            self.controller.set_goal(
                action=np.zeros_like(action),
                set_qpos=action,
            )

        action = self.controller.run_controller()
        self.sim.data.ctrl[:] = action

    def reset(self):
        super().reset()
        if self.controller is not None:
            self.controller.reset_goal()

    def sample_init_state(self):
        # sample a value samller than the upper bound and larger than the lower bound
        x_limits, y_limist = (
            self.cfg.builder_cfg.pusher.x_limits,
            self.cfg.builder_cfg.pusher.y_limits,
        )

        sample_value = lambda lower, upper: lower + (upper - lower) * np.random.rand()

        pusher_x_value = sample_value(x_limits[0], x_limits[1])
        pusher_y_value = sample_value(y_limist[0], y_limist[1])

        self.sim.data.set_joint_qpos("pusher_x_joint", pusher_x_value)
        self.sim.data.set_joint_qpos("pusher_y_joint", pusher_y_value)

        if self.object_sampler is not None:
            object_placements = self.object_sampler.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        # call forward to let the new state take effect
        self.forward()

    def sample_trajectory(self, horizon=100, get_obs=None):
        if any([x.name == "rod" for x in self.cfg.builder_cfg.objects]):
            raise ValueError(
                """
                Does not support trajectory sampling with rod objects.
                Currently the pusher will push the object out of bounds.
                """
            )

        self.sample_init_state()

        # let the object settle down, just in case they are in the air
        for _ in range(50):
            self.sim.step()

        random_object_index = random.randint(0, len(self.mj_objects) - 1)

        position_cmd = np.array(
            [np.random.random() * 1.5 - 1.0, np.random.random() * 2 - 1.0, 0]
        )

        for i in range(horizon):
            if len(self.mj_objects) > 0:
                position_cmd = self.data.get_body_xpos(
                    f"object_{random_object_index}_main"
                )

            curr_pos = self.data.get_body_xpos(f"pusher_main")

            random_cmd = position_cmd - curr_pos
            random_cmd = (random_cmd / np.linalg.norm(random_cmd)) * 2.0
            random_cmd = np.clip(random_cmd, -1.0, 1.0)

            self.step(action=random_cmd[:2])

            if get_obs is not None:
                get_obs(self)

    def down_traj(self, horizon=50, get_obs=None):
        # reset and generate a traj.
        self.reset()
        self.controller.interpolator.total_steps = 32 * 2

        position_based_command = np.array([0.0, -0.82])
        for i in range(horizon):
            self.step(position_based_command)

            if get_obs is not None:
                get_obs(self)

    def right_traj(self, horizon=50, get_obs=None):
        # reset and generate a traj.
        self.reset()
        self.controller.interpolator.total_steps = 32 * 2

        position_based_command = np.array([0.82, 0.0])
        for i in range(horizon):
            self.step(position_based_command)

            if get_obs is not None:
                get_obs(self)

    def up_traj(self, horizon=50, get_obs=None):
        # reset and generate a traj.
        self.reset()
        self.controller.interpolator.total_steps = 32 * 2

        position_based_command = np.array([0.0, 0.52])
        for i in range(horizon):
            self.step(position_based_command)

            if get_obs is not None:
                get_obs(self)

    def left_traj(self, horizon=50, get_obs=None):
        # reset and generate a traj.
        self.reset()
        self.controller.interpolator.total_steps = 32 * 2

        position_based_command = np.array([-0.52, 0.0])
        for i in range(horizon):
            self.step(position_based_command)

            if get_obs is not None:
                get_obs(self)

    def convert_local_command_to_global(self, du):
        # curr_pos = self.data.get_body_xpos(f"pusher_main")[:2]
        self.controller.update()
        curr_pos = self.controller.joint_pos.copy()
        assert len(du) == len(curr_pos), "shape must be the same"
        next_pos = curr_pos + du

        return next_pos

    def render_segmentation(self, segmentation_type="pusher"):
        mask = self.render("birdview", render_segmentation=True)

        if segmentation_type == "pusher":
            pusher_idx = self.sim.model.geom_name2id("pusher_g0_vis")
            mask = mask[..., 1] == pusher_idx

        elif segmentation_type == "object":
            obj_idx = self.sim.model.geom_name2id("object_0_g0_visual")
            mask = mask[..., 1] == obj_idx

        return mask
