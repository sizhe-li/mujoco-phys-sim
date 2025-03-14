import os
from mujoco_sim.utils.camera_utils import *


def save_cam_params_mj(env, dump_dir):
    camID_to_cam2world = []
    camID_to_intrinsics = []

    for cam_id, view_name in enumerate(env.cam_views):
        cam2world = get_camera_extrinsic_matrix(env, view_name)
        intrinsic = get_camera_intrinsic_matrix(env, view_name)

        camID_to_cam2world.append(cam2world)
        camID_to_intrinsics.append(intrinsic)

    camID_to_cam2world = np.stack(camID_to_cam2world, axis=0)
    camID_to_intrinsics = np.stack(camID_to_intrinsics, axis=0)

    camera_data = {
        'camID_to_cam2world': camID_to_cam2world.astype(np.float32),
        'camID_to_intrinsics': camID_to_intrinsics.astype(np.float32),
    }

    np.savez_compressed(os.path.join(dump_dir, 'camera_config.npz'), **camera_data)


def save_cam_params_raw(*, dump_dir, h, w, fovy, camera_positions, camera_orientations):
    camID_to_cam2world = []
    camID_to_intrinsics = []

    for cam_pos, cam_rot in zip(camera_positions, camera_orientations):
        cam2world = get_camera_extrinsic_matrix_raw(cam_pos, cam_rot)
        intrinsic = get_camera_intrinsic_matrix_raw(h, w, fovy)

        camID_to_cam2world.append(cam2world)
        camID_to_intrinsics.append(intrinsic)

    camID_to_cam2world = np.stack(camID_to_cam2world, axis=0)
    camID_to_intrinsics = np.stack(camID_to_intrinsics, axis=0)

    camera_data = {
        'camID_to_cam2world': camID_to_cam2world.astype(np.float32),
        'camID_to_intrinsics': camID_to_intrinsics.astype(np.float32),
    }

    np.savez_compressed(os.path.join(dump_dir, 'camera_config.npz'), **camera_data)


def parse_spherical_camera_params(spherical_cfg):
    n_views = spherical_cfg.num_views
    radius = spherical_cfg.radius
    cam_height = spherical_cfg.cam_height

    # Set the center of the circular camera motion
    lookat = np.array(spherical_cfg.lookat)
    deg_seq = np.linspace(0., 360, n_views + 1)[:-1]

    camera_positions = []
    camera_orientations = []

    for curr_deg in deg_seq:
        cam_pos = np.array([*point_on_circle(np.deg2rad(curr_deg), radius, center=lookat[:2]), cam_height])
        cam_pos, cam_ori = camera_lookat(cam_pos, lookat, up=np.array([0, 0, 1]), return_quat=False)
        camera_positions.append(cam_pos)
        camera_orientations.append(cam_ori)

    return camera_positions, camera_orientations


def parse_base_camera_params(base_cfg):
    cam_pos, cam_ori = camera_lookat(
        np.array(base_cfg.cam_pos), np.array(base_cfg.lookat),
        up=np.array([0, 0, 1]), return_quat=False
    )

    return cam_pos, cam_ori


def parse_camera_params(view_cfg):
    camera_positions, camera_orientations = [], []

    spherical_params = view_cfg.get('spherical', None)
    if spherical_params is not None:
        _positions, _orientations = parse_spherical_camera_params(spherical_params)

        camera_positions.extend(_positions)
        camera_orientations.extend(_orientations)

    base_params = view_cfg.get('base', None)
    if base_params is not None:
        _pos, _ori = parse_base_camera_params(base_params)
        camera_positions.append(_pos)
        camera_orientations.append(_ori)

    if len(camera_positions) == 0:
        raise ValueError("No camera is found in the configuration file!")


    return camera_positions, camera_orientations
