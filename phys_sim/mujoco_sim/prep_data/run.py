import os
import hydra
import random
from omegaconf import DictConfig
from omegaconf import OmegaConf
import glob
import numpy as np
from tqdm import tqdm
from clean_disentangle.utils.io_utils import (
    PROJ_DIR,
    generate_random_uuid,
    create_folder,
)
from mujoco_sim.utils.camera_utils import *

from . import parse_camera_params, save_cam_params_raw
from mujoco_sim.env.base_env import RobotEnv
from mujoco_sim.env.gripper_env import FrankaEnv, UR5Env


class DataGeneratorBase:
    def __init__(self, cfg):
        self.cfg = cfg

        self.mode = cfg.get("mode", "normal")
        if self.mode == "normal":
            self._sample = self.sample_sim_normal
        elif self.mode == "expand":
            self._sample = self.sample_sim_expand
            assert self.cfg.get("base_dir", None) is not None
        else:
            raise NotImplementedError()

        self.dump_dir = os.path.join(PROJ_DIR, cfg.dump_dir)

        self.debug = cfg.debug
        self.num_samples = cfg.num_samples
        self.remove_exists = cfg.remove_exists

        self.existing_files = []
        self.camera_positions, self.camera_orientations = [], []

        # expand related config
        self.curr_joint_pos = None
        self.curr_atpt_cnt = 1
        self.curr_succ_cnt = 1
        self.max_atpt_per_file = 25000
        self.max_succ_per_file = 5

        self.init_log()
        self.env = self.init_env()
        self.init_camera_params()

    def init_log(self):

        create_folder(self.dump_dir, remove_exists=self.remove_exists)
        if self.mode == "expand":
            base_dir = os.path.join(PROJ_DIR, self.cfg.base_dir)
            assert os.path.isdir(base_dir)
            self.existing_files = glob.glob(os.path.join(base_dir, "*.npz"))
            self.existing_files = [
                x for x in self.existing_files if "camera_config" not in x
            ]

    def init_env(self) -> RobotEnv:
        raise NotImplementedError()

    def init_camera_params(self):
        self.camera_positions, self.camera_orientations = parse_camera_params(
            self.cfg.views
        )

        print("saving camera parameters...")

        img_height, img_width = self.env.img_height, self.env.img_width
        fovy = self.env.mj_model.cam_fovy[0]

        save_cam_params_raw(
            dump_dir=self.dump_dir,
            h=img_height,
            w=img_width,
            fovy=fovy,
            camera_positions=self.camera_positions,
            camera_orientations=self.camera_orientations,
        )

        # change format to fit mujoco after saving
        self.camera_orientations = [
            mat2quat(cam_ori) for cam_ori in self.camera_orientations
        ]

    def sample_sim(self):
        env = self.env

        joint_pos = self._sample()

        # reject if penetrates
        if (
            env.data.contact.dist.shape == (0,)
            or env.data.contact.dist.min() < -2e-4
            or env.data.site(0).xpos[2] < 0.135
        ):
            print("caught penetration! skipped!")
            return None
        else:  # success
            if self.mode == "expand":
                self.curr_succ_cnt += 1
                print(f"[Success {self.curr_succ_cnt} / {self.max_succ_per_file}]")

        return joint_pos

    def sample_sim_normal(self) -> np.array:
        raise NotImplementedError()

    def sample_sim_expand(self) -> np.array:
        raise NotImplementedError()

    def generate_data(self):
        # start sampling
        sample_cnt = 0
        pbar = tqdm(total=self.num_samples)

        env = self.env
        camera_positions = self.camera_positions
        camera_orientations = self.camera_orientations

        use_depth = self.cfg.sensor.depth

        while sample_cnt < self.num_samples:

            joint_pos = self.sample_sim()
            if joint_pos is None:  # failed
                continue

            images = [
                env.render_given_view(cam_pos, cam_ori, render_depth=use_depth)
                for cam_pos, cam_ori in zip(camera_positions, camera_orientations)
            ]

            if use_depth:
                rgb_images = np.stack([x[0] for x in images], axis=0)
                depth_images = np.stack([x[1] for x in images])
            else:
                rgb_images = np.stack(images, axis=0)
                depth_images = None

            data_dict = {
                "image_obs": rgb_images.astype(np.uint8),
                "joint_pos": joint_pos.astype(np.float32),
            }

            if use_depth:
                data_dict["depth_obs"] = depth_images.astype(np.float32)

            save_path = os.path.join(self.dump_dir, f"{generate_random_uuid()}.npz")
            np.savez_compressed(save_path, **data_dict)

            pbar.update(1)
            pbar.set_description(f"Generating box scene:", refresh=True)

            if self.debug:
                print("[Exiting...] Debug mode is ON")
                return

            sample_cnt += 1

        pbar.close()


class UR5DataGenerator(DataGeneratorBase):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init_env(self) -> RobotEnv:
        # initialize env
        env = UR5Env()
        env.set_ctrl_mode("relative")
        # warmup (mj_forward has bugs otherwise)
        env.set_state(env.init_state)
        env.data.ctrl = [0.0] * 6
        for _ in range(20):
            env._sim_step()

        return env

    def sample_sim_normal(self):
        ctrl_range = self.env.mj_model.actuator_ctrlrange
        low_range = ctrl_range[:, 0]
        delta_range = ctrl_range[:, 1] - ctrl_range[:, 0]

        weights = np.random.random((6,))

        self.env.data.qpos[:6] = (low_range + weights * delta_range)[:6]
        self.env.forward()
        joint_pos = self.env.data.qpos[:6]

        return joint_pos

    def sample_sim_expand(self):
        # sample last three degrees of freedom
        sample_inds = [-3, -2, -1]

        if (
            self.curr_joint_pos is None
            or self.curr_atpt_cnt % self.max_atpt_per_file == 0
            or self.curr_succ_cnt % self.max_succ_per_file == 0
        ):
            print(
                f"[Patience: {self.curr_atpt_cnt} / {self.max_atpt_per_file}] sampling new file!"
            )

            # patience is exhausted or joint pos is not set
            base_file = random.choice(self.existing_files)
            base_joint_pos = np.load(base_file)["joint_pos"]

            # change current joint pos
            self.curr_joint_pos = base_joint_pos
            # clear counters
            self.curr_atpt_cnt = 1
            self.curr_succ_cnt = 1
        else:
            # continue sampling from current joint pos
            base_joint_pos = self.curr_joint_pos
            self.curr_atpt_cnt += 1

        print(
            f"[Patience: {self.curr_atpt_cnt} / {self.max_atpt_per_file}] attempting! "
        )

        ctrl_range = self.env.mj_model.actuator_ctrlrange
        low_range = ctrl_range[:, 0]
        delta_range = ctrl_range[:, 1] - ctrl_range[:, 0]

        weights = np.random.random((6,))
        base_joint_pos[sample_inds] = (low_range + weights * delta_range)[sample_inds]

        self.env.data.qpos[:6] = base_joint_pos
        self.env.forward()
        joint_pos = self.env.data.qpos[:6]

        return joint_pos


@hydra.main(
    version_base=None, config_path="../../../assets/config/dataset", config_name="ur5e"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.robot.type == "ur5e":
        gen = UR5DataGenerator(cfg)
    elif cfg.robot.type == "franka":
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    gen.generate_data()


if __name__ == "__main__":
    main()
