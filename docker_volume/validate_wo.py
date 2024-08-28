import hydra
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from client.src.inference_server_client_wo import InferenceServerClient
from dataset.utils import load_dataset_language

from simulation.environments.ur10e_cell import UR10ECell
from simulation.tasks import utils
from simulation.tasks.picking_google_objects import PickSeenGoogleObjects


def read_sample_at_i_and_request(cfg, i: int = 0) -> np.ndarray:
    test_dataset = load_dataset_language(
        50, "/home/robot/shared_docker_volume/storage/data/language/simple/test/")

    gt_pose = test_dataset.datasets['grasp_pose'].read_sample(i)['grasp_pose']
    task_info = test_dataset.datasets['info'].read_sample(i)

    colors = []
    extrinsics = []
    intrinsics = []
    for idx in range(3):
        color = test_dataset.datasets['color'].read_sample_at_idx(
            i, idx)[..., :3]
        camera_config = test_dataset.datasets['camera_config'].read_sample_at_idx(
            i, idx)
        extrinsic = camera_config['pose'].astype(dtype=np.float32)
        intrinsic = camera_config['intrinsics'].astype(dtype=np.float32)
        colors.append(color)
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)
    plt.imshow(colors[0])

    inference_server_client = InferenceServerClient(
        url="http://172.20.1.3:31708")
    optimized_pose, trajectory = inference_server_client.optimize_pose(camera_color_imgs=colors,
                                                                       camera_pose_htms=extrinsics,
                                                                       camera_instrinsics=intrinsics,
                                                                       optimization_config=cfg.optimization_config,
                                                                       reset_optimizer=True)

    pos = optimized_pose[:3, 3]
    gt_pos = gt_pose[:3, 3]
    xyz_off = gt_pos - pos
    print(f'Result: \n {optimized_pose}')
    print(f"Expected: \n {gt_pose}")
    print("Off by: ", xyz_off)

    rot = utils.mat_to_quat(optimized_pose[:3, :3])
    gt_rot = utils.mat_to_quat(gt_pose[:3, :3])
    pose = (pos, rot)
    gt_pose = (gt_pos, gt_rot)
    return pose, gt_pose, task_info, trajectory


def show_marker(env, pose):
    env.add_object(urdf='util/coordinate_axes.urdf',
                   pose=pose, category='fixed')


def create_environment(task_info):
    assets_root = "/home/robot/docker_volume/simulation/assets/"
    env = UR10ECell(
        assets_root=assets_root,
        disp=True,
        hz=480,
        record_cfg=None
    )

    task = PickSeenGoogleObjects(info=task_info)
    task.mode = 'test'
    n_perspectives = 50

    env.set_task(task)
    env.reset()
    return env


@hydra.main(version_base=None, config_path="./client/src/configs", config_name="language_1_view")
def main(cfg):
    show_pos = True
    i = 1

    trajectories = []
    op_poses = []
    for n in range(100):
        print(f"Optimization {n+1} of 250")
        tra = []
        optimized_pose, gt_pose, task_info, trajectory = read_sample_at_i_and_request(
            cfg, i)
        for step in trajectory:
            pose = np.frombuffer(step, dtype=np.float32)
            pose = np.reshape(pose, (4, 4))
            pos = pose[:3, 3]
            rot = utils.mat_to_quat(pose[:3, :3])
            pose = (pos, rot)
            tra.append(pose)
        trajectories.append(tra)
        op_poses.append(optimized_pose)

    env = create_environment(task_info)
    if show_pos:
        for pose in op_poses:
            show_marker(env, pose)
    else:
        show_marker(env, gt_pose)
        for trajectory in trajectories:
            color = [random.random(), random.random(), random.random()]
            for idx in range(len(trajectory)):
                if idx == 0:
                    pass
                else:
                    p.addUserDebugLine(
                        trajectory[idx-1][0], trajectory[idx][0], color)
        p.addUserDebugPoints(
            [trajectory[-1][0]], [color], 3.5)

    # Validate task is spawned correctly
    plt.show()

    while True:
        qKey = ord('q')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
            break

        env.step_simulation()

        # env.step([(optimized_pose[0], optimized_pose[1])])


if __name__ == "__main__":
    main()
