"""
adapted from: https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/utils/camera_utils.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from transforms3d.quaternions import mat2quat


def get_camera_intrinsic_matrix(env, camera_name):
    """
    Obtains camera intrinsic matrix.
    Args:
        env (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    import mujoco

    camera_height = env.img_height
    camera_width = env.img_width

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = env.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(env, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    Args:
        env (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    import mujoco

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera_pos = env.data.cam_xpos[cam_id]
    camera_rot = env.data.cam_xmat[cam_id].reshape(3, 3)
    R = make_pose(camera_pos, camera_rot)

    return R


def get_camera_intrinsic_matrix_raw(camera_height, camera_width, fovy):
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix_raw(camera_pos, camera_rot):
    R = make_pose(camera_pos, camera_rot)

    return R


def camera_lookat(pos, lookat, up=None, return_quat=True):
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos)
    if not isinstance(lookat, np.ndarray):
        lookat = np.array(lookat)
    # Compute the forward direction
    if up is None:
        up = np.array([0., 0., 1.0])
    z = pos - lookat
    z /= np.linalg.norm(z)

    # Compute the up direction
    x = np.cross(up, z)
    x /= np.linalg.norm(x)

    # Compute the right direction
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    # Compute the rotation matrix
    R = np.stack((x, y, z), axis=-1)

    orientation = R
    if return_quat:
        orientation = mat2quat(R)
        # orientation = np.array([qx, qy, qz, qw])

    return pos, orientation


def point_on_circle(angle, radius, center):
    x = radius * np.cos(angle) + center[0]
    y = radius * np.sin(angle) + center[1]
    return np.array([x, y])


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled,
                                1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                  [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
                   vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()


def get_spherical_camera_params(
        img_height,
        img_width,
        fovy=45.0,
        n_views=100,
        radius=2.0,
        cam_height=1.5,
        lookat=None,
):
    """
    :param img_height: image height
    :param img_width: image width
    :param fovy: focal length
    :param n_views: number of views to generate
    :param radius: radius of the sphere surrounding the lookat point
    :param cam_height: height of the camera
    :param lookat: the lookat position of the camera
    :return: dictionary that contains id_to_cam2world, id_to_intrinsics, id_to_campos, id_to_camori
    """

    if lookat is None:
        lookat = [0.0, 0.0, 0.5]

    # Set the center of the circular camera motion
    lookat = np.array(lookat)
    deg_seq = np.linspace(0., 360, n_views + 1)[:-1]

    id_to_campos = []
    id_to_camori = []

    for curr_deg in deg_seq:
        cam_pos = np.array([*point_on_circle(np.deg2rad(curr_deg), radius, center=lookat[:2]), cam_height])
        cam_pos, cam_ori = camera_lookat(cam_pos, lookat, up=np.array([0, 0, 1]), return_quat=False)
        id_to_campos.append(cam_pos)
        id_to_camori.append(cam_ori)

    id_to_cam2world = []
    id_to_intrinsics = []

    for cam_pos, cam_rot in zip(id_to_campos, id_to_camori):
        cam2world = get_camera_extrinsic_matrix_raw(cam_pos, cam_rot)
        intrinsic = get_camera_intrinsic_matrix_raw(img_height, img_width, fovy)

        id_to_cam2world.append(cam2world)
        id_to_intrinsics.append(intrinsic)

    id_to_cam2world = np.stack(id_to_cam2world, axis=0)
    id_to_intrinsics = np.stack(id_to_intrinsics, axis=0)

    id_to_camori = [mat2quat(cam_ori) for cam_ori in id_to_camori]

    output = {'id_to_cam2world': id_to_cam2world,
              'id_to_intrinsics': id_to_intrinsics,
              'id_to_campos': id_to_campos,
              'id_to_camori': id_to_camori}

    return output


def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.
    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation
    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose
