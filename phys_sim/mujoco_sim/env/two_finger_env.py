from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Literal
from xml.etree import ElementTree
import copy

import mujoco
from mujoco_sim.models.base import MujocoXML
from robosuite.utils.mjcf_utils import xml_path_completion
from mujoco_sim.env.base_env import RobotEnv, RobotEnvCfg
from robosuite.models.objects import BoxObject, CylinderObject, MujocoObject
from robosuite.controllers.interpolators.linear_interpolator import LinearInterpolator
from mujoco_sim.models.arenas.empty_arena import EmptyArena
from robosuite.utils.placement_samplers import UniformRandomSampler
from mujoco_sim.models import MujocoWorldBase
from robosuite.utils.mjcf_utils import (
    CustomMaterial,
    add_material,
    ALL_TEXTURES,
    new_geom,
)
import numpy as np
from mujoco_sim.utils import xml_utils
from robosuite.utils.binding_utils import (
    MjData,
    MjModel,
    MjRenderContextOffscreen,
    MjSim,
    MjSimState,
)
from robosuite.utils.camera_utils import get_real_depth_map

from .push_env import BoxCfg, RodCfg, Position_To_String, MeshObject, MaterialCfg

# root_dir = Path("/home/iyu/scene-jacobian-discovery/assets/mujoco/")

root_dir = Path(__file__).parent.parent.parent.parent / "assets" / "mujoco"
xml_file = str(root_dir / "planar_hand_reorientation" / "scene_both_fingers.xml")

left_finger_xml = str(root_dir / "planar_hand_reorientation" / "left_finger.xml")

right_finger_xml = str(root_dir / "planar_hand_reorientation" / "right_finger.xml")


@dataclass
class SimulationCfg:
    num_substeps: int
    xml_config_filename: Path | None
    scale: float | None = None


@dataclass
class RenderingCfg:
    cam_views: List[str]
    image_resolution: Tuple[int, int] = (256, 256)


PushEnvObjectCfg = BoxCfg | RodCfg


@dataclass
class TwoFingerEnvBuilderCfg:
    objects: List[PushEnvObjectCfg] = field(default_factory=lambda: [])


@dataclass
class TwoFingerEnvObjectSamplerCfg:
    # use default factory settings
    x_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    y_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])


@dataclass
class TwoFingerEnvCfg(RobotEnvCfg):
    builder_cfg: TwoFingerEnvBuilderCfg
    sampler_cfg: TwoFingerEnvObjectSamplerCfg
    base_xml: str = xml_file
    left_finger_xml: str = left_finger_xml
    right_finger_xml: str = right_finger_xml


class TwoFingerEnvBuilder:

    def __init__(
        self, cfg: TwoFingerEnvBuilderCfg, world: MujocoWorldBase = MujocoWorldBase()
    ):
        self.cfg = cfg
        self.world = world

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

    def build_object(
        self,
        et_name: str,
        object_cfg: PushEnvObjectCfg,
    ) -> Tuple[MujocoObject, ElementTree.Element]:

        if object_cfg.name == "box":
            position_string = Position_To_String(*object_cfg.position)

            material = self.build_material(object_cfg.material)
            object_builder = BoxObject(
                name=et_name,
                size=object_cfg.size,
                rgba=[0, 0.5, 0.5, 1],
                density=object_cfg.density,
                material=material,
                friction=[0.1, 0.1, 0.1],
            )
            object_builder_copy = copy.deepcopy(object_builder)

            object_et = object_builder.get_obj()
            object_et.set("pos", position_string)
            # add a geom to the object
            print("objsize", object_cfg.size)
            object_et.append(
                new_geom(
                    name="obj_x_axis_geom",
                    pos="0.0 0.15 -0.00",
                    axisangle="1 0 0 90",
                    type="cylinder",
                    rgba="0.0 1.0 0.0 0.5",
                    size="0.01 0.10",
                    conaffinity="0",
                    contype="0",
                    group="1",
                )
            )
            object_et.append(
                new_geom(
                    name="obj_y_axis_geom",
                    pos="-0.15 0.0 -0.0",
                    axisangle="0 1 0 90",
                    type="cylinder",
                    rgba="1.0 0.0 0.0 0.5",
                    size="0.01 0.10",
                    conaffinity="0",
                    contype="0",
                    group="1",
                )
            )

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
            print(f"object_et: {object_et}", position_string)
            object_et.set("pos", position_string)

        return object_builder_copy, object_et

    def build_objects(self):
        object_cfgs = self.cfg.objects
        print(f"building {len(object_cfgs)} objects")
        mujoco_objects = []

        for idx, object_cfg in enumerate(object_cfgs):
            et_name = f"object_{idx}"
            obj_builder, obj_et = self.build_object(et_name, object_cfg)

            mujoco_objects.append(obj_builder)
            self.world.worldbody.append(obj_et)
            self.world.merge_assets(obj_builder)
            print("bodies", self.world.worldbody)
            print(f"built object {idx}")

        return mujoco_objects

    def build(self) -> Tuple[mujoco.MjModel, List[MujocoObject]]:
        mujoco_objects = self.build_objects()

        model = self.world.get_model(mode="mujoco")

        return model, mujoco_objects


class TwoFingerEnv(RobotEnv):
    cfg: TwoFingerEnvCfg
    qpos_limits = np.array(
        [
            # left finger
            [-2.1, 0],  # min, max for the first joint
            [-2.1, 0],  # min, max for the second joint
            # right finger
            [0, 2.1],
            [0, 2.1],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        cfg: TwoFingerEnvCfg,
        mj_model: MjModel | None,
    ):
        self.cfg = cfg
        self.sampler_cfg = cfg.sampler_cfg

        self.world = MujocoXML(cfg.base_xml)
        self.world.merge(MujocoXML(cfg.left_finger_xml))
        self.world.merge(MujocoXML(cfg.right_finger_xml))

        builder = TwoFingerEnvBuilder(cfg.builder_cfg, self.world)
        self.mj_model, self.objects = builder.build()

        if len(self.objects) > 0:
            self.object_sampler = UniformRandomSampler(
                name="sampler",
                mujoco_objects=self.objects,
                x_range=self.sampler_cfg.x_range,
                y_range=self.sampler_cfg.y_range,
            )
        else:
            self.object_sampler = None

        # detect if there is an object named 'box' in the model
        super().__init__(cfg=cfg, mj_model=self.mj_model)

        self.controller = None

    def _pre_action(self, action):
        self.sim.data.ctrl[:] = action
        # clip the action to the joint limits
        self.sim.data.ctrl[:] = np.clip(
            self.sim.data.ctrl[:],
            self.sim.model.actuator_ctrlrange[:, 0],
            self.sim.model.actuator_ctrlrange[:, 1],
        )

    def step(self, action: np.ndarray):
        self._pre_action(action)

        for i in range(self.num_substeps):
            self.sim.forward()
            self.sim.step()

    def _is_colliding(self):
        # make sure the object is not in collision with the fingers
        return (
            self.sim.data.contact.dist.shape == (0,)
            or self.sim.data.contact.dist.min() < -2e-4
        )

    def sample_init_state(self):
        # sample a value samller than the upper bound and larger than the lower bound
        qpos = np.random.uniform(self.qpos_limits[:, 0], self.qpos_limits[:, 1])

        self.step(qpos)

        if self.object_sampler is not None:
            object_placements = self.object_sampler.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        # call forward to let the new state take effect
        self.forward()

        not_colliding = not self._is_colliding()

        return not_colliding

    def convert_local_command_to_global(self, du: np.ndarray) -> np.ndarray:
        curr_pos = np.array(self.data.ctrl[:])

        next_pos = curr_pos + du
        next_pos = np.clip(
            next_pos,
            self.sim.model.actuator_ctrlrange[:, 0],
            self.sim.model.actuator_ctrlrange[:, 1],
        )

        return next_pos

    def reset(self):
        super().reset()
        if self.controller is not None:
            self.controller.reset_goal()
