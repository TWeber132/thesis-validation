import hydra
import numpy as np

from client.src.inference_server_client import InferenceServerClient
from dataset.utils import load_dataset_language

from simulation.environments.environment import Environment
# from simulation.tasks.picking_google_objects import PickSeenGoogleObjects


def read_sample_at_i_and_request(cfg, i: int = 0) -> np.ndarray:
    test_dataset = load_dataset_language(
        cfg.dataset.n_perspectives, cfg.dataset.path + '/test')

    i = 0
    text_query = test_dataset.datasets['language'].read_sample(i)
    gt_pose = test_dataset.datasets['grasp_pose'].read_sample(i)['grasp_pose']
    task_info = test_dataset.datasets['task_info'].read_sample(i)

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

    inference_server_client = InferenceServerClient(
        url="http://172.20.1.3:31708")
    optimized_pose = inference_server_client.optimize_pose(camera_color_imgs=colors,
                                                           camera_pose_htms=extrinsics,
                                                           camera_instrinsics=intrinsics,
                                                           optimization_config=cfg.optimization_config,
                                                           text_query=text_query,
                                                           reset_optimizer=True)
    op_xyz = optimized_pose[:3, 3]
    gt_xyz = gt_pose[:3, 3]
    xyz_off = gt_xyz - op_xyz
    print(f'Result: \n {optimized_pose}')
    print(f"Expected: \n {gt_pose}")
    print("Off by: ", xyz_off)
    return optimized_pose, task_info


def show_marker(env, pose):
    env.add_object(urdf='util/coordinate_axes.urdf',
                   pose=pose, category='fixed')


def create_environment():
    env = Environment(
        assets_root="/home/robot/docker_volume/simulation/assets/",
        disp=True,
        hz=480,
        record_cfg=None
    )
    return env
    # task = PickSeenGoogleObjects
    # task.mode = 'test'
    # n_perspectives = 50


@hydra.main(version_base=None, config_path="./configs", config_name="language_1_view")
def main(cfg):
    optimized_pose, task_info = read_sample_at_i_and_request(cfg, i=0)
    print(task_info)
    env = create_environment()
    show_marker(env, optimized_pose)
    env.step([(optimized_pose[0], optimized_pose[1])])


if __name__ == "__main__":
    main()
