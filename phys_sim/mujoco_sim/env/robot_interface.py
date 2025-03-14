from typing import List, Tuple

UR5E_MIN_QPOS = [-6.2831, -3.1415, -3.1415, -6.2831, -6.2831, -6.2831]
UR5E_MAX_QPOS = [6.2831, 0., 3.1415, 6.2831, 6.2831, 6.2831]

PANDA_MIN_QPOS = [-2.897, -1.763, -2.897, -3.072, -2.897, -0.018, -2.897, 0.]
PANDA_MAX_QPOS = [2.897, 1.763, 2.897, -0.07, 2.897, 3.752, 2.897, 0.04]

PUSHER_MIN_QPOS = [-1.7, -1.09]
PUSHER_MAX_QPOS = [0.4, 1.0]

supported_robots = ["ur5e", "panda", "pusher"]


def get_robot_joint_range(robot_name: str) -> Tuple[List[float], List[float]]:
    if robot_name == 'ur5e':
        min_qpos = UR5E_MIN_QPOS
        max_qpos = UR5E_MAX_QPOS
    elif robot_name == 'panda':
        min_qpos = PANDA_MIN_QPOS
        max_qpos = PANDA_MAX_QPOS
    elif robot_name == "pusher":
        min_qpos = PUSHER_MIN_QPOS
        max_qpos = PUSHER_MAX_QPOS
    else:
        raise ValueError(f'Unsupported robot. name: {robot_name}')

    return min_qpos, max_qpos
