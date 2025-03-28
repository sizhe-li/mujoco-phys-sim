from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import mujoco
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

map_to_root = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class SimulationCfg:
    num_substeps: int
    xml_config_filename: Path | None
    scale: float | None = None


@dataclass
class RenderingCfg:
    cam_views: List[str]
    image_resolution: Tuple[int, int] = (256, 256)


@dataclass
class RobotEnvCfg:
    simulation: SimulationCfg
    rendering: RenderingCfg


class RobotEnv:
    cfg: RobotEnvCfg
    mj_model: MjModel
    sim: MjSim
    data: MjData
    init_state: MjSimState
    num_substeps: int

    def __init__(
        self,
        cfg: RobotEnvCfg,
        mj_model: MjModel | None,
    ):
        super(RobotEnv, self).__init__()
        self.cfg = cfg

        self._load_mj_model(
            mj_model,
            cfg.simulation.xml_config_filename,
            scale=cfg.simulation.scale,
        )
        self._init_sim()
        self._init_renderer()
        self._reset_internal()

    def _load_mj_model(
        self,
        mj_model: MjModel | None,
        xml_config_filename: Path | None,
        scale: float | None = None,
    ):

        if mj_model is not None:
            print("Loading input MjModel")
            self.mj_model = mj_model

        elif xml_config_filename is not None:
            xml_config_filename = map_to_root / xml_config_filename
            print(f"Loading XML file {xml_config_filename}")
            self.mj_model = xml_utils.load_mj_model_from_xml(
                str(xml_config_filename), scale=scale
            )

        else:
            raise NotImplementedError(
                "Must provide either model or xml file to initialize!"
            )

    def _init_sim(self):
        self.sim = MjSim(self.mj_model)
        self.data = self.sim.data

        self.init_state = self.sim.get_state()
        self.num_substeps = self.cfg.simulation.num_substeps

    def _init_renderer(self):
        render_ctxt = MjRenderContextOffscreen(self.sim, device_id=-1)
        self.sim.add_render_context(render_ctxt)

    def _reset_internal(self):
        self.sim.set_state(self.init_state)

    def set_state(self, state):
        self.sim.set_state(state)

    def get_state(self):
        return self.sim.get_state()

    def reset(self):
        self._reset_internal()
        self.sim.forward()

    def forward(self):
        self.sim.forward()

    def step(self, action: np.ndarray, policy_step: bool = True):

        policy_step = True
        for i in range(self.num_substeps):
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            policy_step = False

    def _pre_action(self, action, policy_step=False):
        raise NotImplementedError

    def render(
        self,
        camera_name: str | None = None,
        render_depth: bool = False,
        render_segmentation: bool = False,
    ):
        if camera_name is None:
            camera_name = self.sim.model.camera_id2name(0)

        render_outputs = self.sim.render(
            camera_name=camera_name,
            height=self.cfg.rendering.image_resolution[0],
            width=self.cfg.rendering.image_resolution[1],
            depth=render_depth,
            segmentation=render_segmentation,
        )

        if isinstance(render_outputs, tuple):
            render_outputs = tuple(np.flip(x, axis=0) for x in render_outputs)
        else:
            render_outputs = np.flip(render_outputs, axis=0)

        if render_depth:
            x, render_depth = render_outputs
            render_depth = get_real_depth_map(self.sim, render_depth)
            render_outputs = (x, render_depth)

        return render_outputs

    def render_multiview(self):
        assert len(self.cfg.rendering.cam_views) > 0

        images = []
        for view_name in self.cfg.rendering.cam_views:
            images.append(self.render(view_name))

        return images

    def render_given_view(
        self,
        cam_pos: np.ndarray,
        cam_ori: np.ndarray,
        view_name: str | None = None,
        render_depth: bool = False,
        render_segmentation: bool = False,
    ):
        if view_name is None:
            view_name = self.sim.model.camera_id2name(0)

        self.mj_model.camera(view_name).pos = cam_pos
        self.mj_model.camera(view_name).quat = cam_ori

        self.sim.forward()
        img = self.render(
            view_name,
            render_depth=render_depth,
            render_segmentation=render_segmentation,
        )

        return img
