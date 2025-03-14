from mujoco_sim.env.push_env import PushEnv
from clean_disentangle.utils.io_utils import PROJ_DIR
import hydra
from omegaconf import DictConfig
import os
import numpy as np
from tqdm import trange

from clean_disentangle.utils.io_utils import (
    PROJ_DIR,
    generate_random_uuid,
    create_folder,
)
from . import save_cam_params_raw


def get_multiview_params():
    from mujoco_sim.utils.camera_utils import get_spherical_camera_params

    cam_params = get_spherical_camera_params(
        256,
        256,
        n_views=8,
        radius=1.5,
        cam_height=2.0,
        lookat=[-0.25, 0.0, 0.0],
    )
    return cam_params


def save_cam_params(dump_dir, cam_params):
    camera_data = {
        "id_to_cam2world": cam_params["id_to_cam2world"].astype(np.float32),
        "id_to_intrinsics": cam_params["id_to_intrinsics"].astype(np.float32),
    }

    np.savez_compressed(os.path.join(dump_dir, "camera_config.npz"), **camera_data)


@hydra.main(
    version_base=None,
    config_path="../../../assets/config/dataset",
    config_name="capsule_only",
)
def main(cfg: DictConfig):
    dump_dir: str = os.path.join(PROJ_DIR, cfg.dump_dir)
    num_trajs: int = cfg.num_trajs
    remove_exists: bool = cfg.remove_exists
    horizon: int = cfg.horizon
    debug: bool = cfg.debug
    use_obj: bool = cfg.use_obj

    # create output folder
    create_folder(dump_dir, remove_exists=remove_exists)

    env = PushEnv(use_object=use_obj)
    pusher_idx = env.sim.model.geom_name2id("pusher_g0_vis")

    cam_params = get_multiview_params()
    # save nerf format cam params
    save_cam_params(dump_dir, cam_params)
    # grab mujoco format cam params
    id_to_campos, id_to_camori = cam_params["id_to_campos"], cam_params["id_to_camori"]

    obs_keys = ("birdview_rgb", "birdview_mask", "multiview_rgb", "joint_pos")
    observations = {k: [] for k in obs_keys}

    def render_birdview(env: PushEnv):
        rgb = env.render("birdview", segmentation=False)
        mask = env.render("birdview", segmentation=True)

        mask = mask[..., 1] == pusher_idx

        return rgb, mask

    def render_multiview(env: PushEnv):
        multiview_obs = []
        for cam_pos, cam_ori in zip(id_to_campos, id_to_camori):
            rgb = env.render_given_view(cam_pos, cam_ori, render_segmentation=False)
            multiview_obs.append(rgb)
        multiview_obs = np.stack(multiview_obs, axis=0)
        return multiview_obs

    def get_obs(env: PushEnv):
        birdview_rgb, birdview_mask = render_birdview(env)
        multiview_obs = render_multiview(env)
        jpos = env.data.get_body_xpos(f"pusher_main").copy()

        observations["birdview_rgb"].append(birdview_rgb.astype(np.uint8))  # H, W, C
        observations["birdview_mask"].append(birdview_mask.astype(bool))  # H, W
        observations["multiview_rgb"].append(
            multiview_obs.astype(np.uint8)
        )  # N_view, H, W, C
        observations["joint_pos"].append(jpos.astype(np.float32))  # N_q

    def save_traj():
        traj_identifier = generate_random_uuid()

        for k in obs_keys:
            observations[k] = np.stack(observations[k], axis=0)

        # for t in range(horizon):
        #     frame_identifier = f"{t:04d}"
        #
        #     per_frame_obs = dict()
        #     for k in obs_keys:
        #         per_frame_obs[k] = observations[k][t]
        #
        #     save_path = os.path.join(dump_dir, f"{traj_identifier}_{frame_identifier}.npz")

        save_path = os.path.join(dump_dir, f"{traj_identifier}.npz")
        np.savez_compressed(save_path, **observations)

    def clear_obs():
        for k in obs_keys:
            observations[k] = []

    for _ in trange(num_trajs):
        env.sample_rand_traj(horizon=horizon, get_obs=get_obs)
        save_traj()
        clear_obs()

        if debug:
            exit(0)


if __name__ == "__main__":
    main()
