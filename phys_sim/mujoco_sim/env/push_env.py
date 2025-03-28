import copy
import random
from dataclasses import dataclass, field
from typing import List, Literal, Tuple
from xml.etree import ElementTree

import mujoco
import numpy as np
import splipy as sp
import splipy.curve_factory as cf
from mujoco_sim.env.base_env import RobotEnv, RobotEnvCfg
from mujoco_sim.models import MujocoWorldBase
from mujoco_sim.models.arenas.empty_arena import EmptyArena
from mujoco_sim.models.objects import MeshObject
from robosuite.controllers.parts.generic.joint_pos import JointPositionController
from robosuite.models.objects import BoxObject, CylinderObject, MujocoObject
from robosuite.utils.mjcf_utils import CustomMaterial, new_actuator
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.traj_utils import LinearInterpolator
from scipy.stats import norm


Position_To_String = lambda a, b, c: f"{a:.3f} {b:.3f} {c:.3f}"
SupportedLetterNames = Literal["A", "E", "G", "M", "R", "T", "V"]
ObjectTypes = Literal["box", "A", "E", "G", "M", "R", "T", "V", "star", "rod"]
TextureTypes = Literal[
    "WoodRed",
    "WoodGreen",
    "WoodBlue",
    "WoodLight",
    "WoodDark",
    "WoodTiles",
    "WoodPanels",
    "WoodgrainGray",
    "PlasterCream",
    "PlasterPink",
    "PlasterYellow",
    "PlasterGray",
    "PlasterWhite",
    "BricksWhite",
    "Metal",
    "SteelBrushed",
    "SteelScratched",
    "Brass",
    "Bread",
    "Can",
    "Ceramic",
    "Cereal",
    "Clay",
    "Dirt",
    "Glass",
    "FeltGray",
    "Lemon",
]


@dataclass
class MaterialCfg:
    texture_type: TextureTypes
    texture_name: str
    material_name: str
    texrepeat: str
    specular: str
    shininess: str


@dataclass
class PusherCfg:
    position: List[float]
    size: List[float]
    rgba: List[float]
    x_limits: List[float]
    y_limits: List[float]
    material: MaterialCfg


@dataclass
class MujocoObjectCfg:
    material: MaterialCfg


@dataclass
class BoxCfg(MujocoObjectCfg):
    name: Literal["box"]
    size: List[float]
    rgba: List[float]
    position: List[float]
    density: float = 1000.0


@dataclass
class LetterCfg(MujocoObjectCfg):
    name: Literal["letter"]
    position: List[float]
    letter_name: SupportedLetterNames


@dataclass
class RodCfg(MujocoObjectCfg):
    name: Literal["rod"]
    position: List[float]


PushEnvObjectCfg = BoxCfg | LetterCfg | RodCfg


@dataclass
class PushEnvBuilderCfg:
    pusher: PusherCfg
    objects: List[PushEnvObjectCfg]


class PushEnvBuilder:
    cfg = PushEnvBuilderCfg

    def __init__(self, cfg: PushEnvBuilderCfg):
        self.cfg = cfg

    def build_material(self, material_cfg: MaterialCfg):

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": material_cfg.texrepeat,
            "specular": material_cfg.specular,
            "shininess": material_cfg.shininess,
        }

        material = CustomMaterial(
            texture=material_cfg.texture_type,
            tex_name=material_cfg.texture_name,
            mat_name=material_cfg.material_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        return material

    def build_pusher(
        self,
        world: MujocoWorldBase,
    ) -> CylinderObject:
        pusher_cfg: PusherCfg = self.cfg.pusher

        box_size = 0.15

        self.pusher_x_lims = (-1.5, 1.5)
        self.pusher_y_lims = (-1.5, 1.5)

        pusher_params = {
            "name": "pusher",
            "size": pusher_cfg.size,
            "rgba": pusher_cfg.rgba,
            "joints": [
                {
                    "name": "pusher_x_joint",
                    "type": "slide",
                    "axis": "1 0 0",
                    "damping": "50",
                    "range": f"{pusher_cfg.x_limits[0]} {pusher_cfg.x_limits[1]}",
                    "limited": "true",
                },
                {
                    "name": "pusher_y_joint",
                    "type": "slide",
                    "axis": "0 1 0",
                    "damping": "50",
                    "range": f"{pusher_cfg.y_limits[0]} {pusher_cfg.y_limits[1]}",
                    "limited": "true",
                },
            ],
        }

        position_string = Position_To_String(*pusher_cfg.position)
        material = self.build_material(pusher_cfg.material)

        pusher = CylinderObject(**pusher_params, material=material)
        pusher_obj = pusher.get_obj()
        pusher_obj.set("pos", position_string)

        world.merge_assets(pusher)
        world.worldbody.append(pusher_obj)

        world.actuator.append(
            new_actuator(
                name="torq_jx",
                joint="pusher_x_joint",
                act_type="motor",
                ctrllimited="true",
                ctrlrange="-135.0 135.0",
            )
        )

        world.actuator.append(
            new_actuator(
                name="torq_jy",
                joint="pusher_y_joint",
                act_type="motor",
                ctrllimited="true",
                ctrlrange="-135.0 135.0",
            )
        )

        return pusher

    def build_object(
        self,
        et_name: str,
        object_cfg: PushEnvObjectCfg,
    ) -> Tuple[MujocoObject, ElementTree.Element]:

        if object_cfg.name == "box":
            position_string = Position_To_String(*object_cfg.position)

            object_builder = BoxObject(
                name=et_name,
                size=object_cfg.size,
                rgba=object_cfg.rgba,
                material=self.build_material(object_cfg.material),
            )
            object_builder_copy = copy.deepcopy(object_builder)

            object_et = object_builder.get_obj()
            object_et.set("pos", position_string)

        elif object_cfg.name == "letter":
            position_string = Position_To_String(*object_cfg.position)

            object_builder = MeshObject(
                name=et_name,
                xml_path=f"objects/letter_{object_cfg.letter_name.upper()}.xml",
            )
            object_builder_copy = copy.deepcopy(object_builder)

            object_et = object_builder.get_obj()
            object_et.set("pos", position_string)

        elif object_cfg.name == "rod":
            position_string = Position_To_String(*object_cfg.position)

            object_builder = MeshObject(name=et_name, xml_path=f"objects/rod.xml")
            object_builder_copy = copy.deepcopy(object_builder)

            object_et = object_builder.get_obj()
            object_et.set("pos", position_string)

        return object_builder_copy, object_et

    def build_objects(self, world: MujocoWorldBase):
        object_cfgs = self.cfg.objects
        mujoco_objects = []

        for idx, object_cfg in enumerate(object_cfgs):
            et_name = f"object_{idx}"
            obj_builder, obj_et = self.build_object(et_name, object_cfg)

            mujoco_objects.append(obj_builder)
            world.worldbody.append(obj_et)
            world.merge_assets(obj_builder)

        return mujoco_objects

    def build(self) -> Tuple[mujoco.MjModel, List[MujocoObject]]:
        world = MujocoWorldBase()
        arena = EmptyArena()
        world.merge(arena)

        self.build_pusher(world)
        mujoco_objects = self.build_objects(world)

        model = world.get_model(mode="mujoco")

        return model, mujoco_objects


@dataclass
class PushEnvObjectSamplerCfg:
    # use default factory settings
    x_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    y_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])


@dataclass
class PushEnvCfg(RobotEnvCfg):
    builder_cfg: PushEnvBuilderCfg
    sampler_cfg: PushEnvObjectSamplerCfg


class PushEnv(RobotEnv):
    cfg: PushEnvCfg
    mj_objects: List[MujocoObject]
    object_sampler: UniformRandomSampler | None

    def __init__(
        self,
        cfg: PushEnvCfg,
    ):
        self.cfg = cfg
        builder_cfg = cfg.builder_cfg
        sampler_cfg = cfg.sampler_cfg

        mj_model, mj_objects = PushEnvBuilder(builder_cfg).build()

        if len(mj_objects) > 0:
            self.object_sampler = UniformRandomSampler(
                name="sampler",
                mujoco_objects=mj_objects,
                x_range=sampler_cfg.x_range,
                y_range=sampler_cfg.y_range,
            )
        else:
            self.object_sampler = None
        self.mj_objects = mj_objects

        super().__init__(cfg=cfg, mj_model=mj_model)

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

    def step_interp(self, control_points: np.ndarray, horizon=10, get_obs=None):
        for i in range(control_points.shape[0] - 1):
            policy_step = True
            for h in range(horizon):
                if h == int(2 * horizon / 3):
                    break
                for j in range(self.num_substeps):
                    t = (1.0 * j) / self.num_substeps
                    self.sim.forward()
                    self._pre_action(
                        (1 - t) * control_points[i] + t * control_points[i + 1],
                        policy_step,
                    )
                    self.sim.step()
                if get_obs is not None:
                    get_obs(self)
        for _ in range(horizon):
            policy_step = True
            for j in range(self.num_substeps):
                self.sim.forward()
                self._pre_action(control_points[-1], policy_step)
                self.sim.step()
                policy_step = False
            if get_obs is not None:
                get_obs(self)

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

        if len(self.mj_objects) > 0:
            random_object_index = random.randint(0, len(self.mj_objects) - 1)
        else:
            random_object_index = 0

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

    def straight_line_traj(
        self, position_based_command, horizon=50, get_obs=None, reset=True
    ):
        if reset:
            self.reset()
        self.controller.interpolator.total_steps = 32 * 2
        for i in range(horizon):
            self.step(position_based_command)

            if get_obs is not None:
                get_obs(self)

    def down_traj(
        self,
        horizon=50,
        get_obs=None,
        reset=True,
        position_based_command=np.array([0.0, -0.5]),
    ):
        # take a photo of the mask
        self.straight_line_traj(position_based_command, horizon, get_obs, reset)

    def right_traj(
        self,
        horizon=50,
        get_obs=None,
        reset=True,
        position_based_command=np.array([0.82, 0.0]),
    ):
        # take a photo of the mask
        self.straight_line_traj(position_based_command, horizon, get_obs, reset)

    def bezier_traj(
        self, curve: cf.bezier, curve_resolution=50, horizon=50, get_obs=None
    ):
        self.reset()
        self.controller.interpolator.total_steps = 1
        t = np.linspace(curve.start()[0], curve.end()[0], curve_resolution)
        # map t to gaussian to make it slow down at the ends
        t_gaussian = norm.cdf(t, 0.5, 0.15)
        t_gaussian = t_gaussian / np.max(t_gaussian)
        curve_points = curve(t)

        # take the diff to get the velocity
        curve_points_du = np.diff(curve_points, axis=0)
        print("curve points", curve_points)
        print("curve points du", curve_points_du)
        for du in curve_points_du:
            robot_cmd = self.convert_local_command_to_global(du)
            for i in range(horizon):
                self.step(robot_cmd)

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

    def random_angled_traj(self, horizon=50, get_obs=None):
        # reset and generate a traj.
        self.reset()
        self.controller.interpolator.total_steps = 32 * 2

        # randomly sample a normalized vector and make sure the max length is 1.0
        position_based_command = np.random.rand(2) * 2 - 1

        for i in range(horizon):
            self.step(position_based_command)

            if get_obs is not None:
                get_obs(self)

    def convert_local_command_to_global(self, du):
        # curr_pos = self.data.get_body_xpos(f"pusher_main")[:2]
        # print("curr pos from xbody", self.data.get_body_xpos("pusher_main")[:2])
        self.controller.update()
        curr_pos = self.controller.joint_pos.copy()
        # print("curr_pos from controller", curr_pos)
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
