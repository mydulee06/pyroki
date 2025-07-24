import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import yaml
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Import functions and classes
from eetrack.utils.weld_objects import WeldObject
import pyroki as pk
import yourdfpy
from pyroki.collision._robot_collision_custom import RobotCollision
import jaxls
from jaxls import TerminationConfig

# --- Utility functions (from 15_mid_sole_sampling_vis.py) ---
def load_config():
    asset_dir = Path(__file__).parent / "eetrack"
    config_file = asset_dir / "config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config, asset_dir

def load_robot(config):
    urdf_path = config['robot']['urdf_path']
    urdf_obj = yourdfpy.URDF.load(urdf_path)
    sit_terminal_states = np.load(config['robot']['sit_terminal_states_path'])
    idx = np.abs(sit_terminal_states["target_height"] - config['robot']['sit_target_height']).argmin()
    joint_pos = sit_terminal_states["joint_state"][idx, 0]
    lab2yourdf = [np.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in urdf_obj.actuated_joint_names]
    urdf_obj.update_cfg(joint_pos[lab2yourdf])
    for joint in urdf_obj.robot.joints:
        if joint.name in urdf_obj.actuated_joint_names and joint.name not in config['robot']['movable_joints']:
            joint.type = "fixed"
            joint.origin = urdf_obj.get_transform(joint.child, joint.parent)
    modified_urdf = yourdfpy.URDF(urdf_obj.robot, mesh_dir=Path(urdf_path).parent)
    collision_cfg = config.get('collision', {})
    ignore_pairs = tuple(tuple(pair) for pair in collision_cfg.get('ignore_pairs', []))
    exclude_links = tuple(collision_cfg.get('exclude_links', []))
    robot_collision = RobotCollision.from_urdf(
        modified_urdf,
        user_ignore_pairs=ignore_pairs,
        ignore_immediate_adjacents=collision_cfg.get('ignore_adjacent_links', True),
        exclude_links=exclude_links
    )
    return pk.Robot.from_urdf(modified_urdf), modified_urdf, robot_collision

def get_welding_object_and_pose(config, modified_urdf, sampled_x=None, sampled_y=None, sampled_yaw=None, sampled_z=None):
    welding_object_config = config["welding_object"].copy()
    welding_object_config.pop('pose', None)
    welding_object_config.pop('yaw', None)
    px = sampled_x or 0.0
    py = sampled_y or 0.0
    pz = sampled_z or 0.0
    yaw = sampled_yaw or 0.0
    so3 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw)
    welding_object_pose_world = jaxlie.SE3.from_rotation_and_translation(so3, jnp.array([px, py, pz]))
    parent = welding_object_config.pop("parent", None)
    if parent == "mid_sole_link":
        left_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("left_sole_link")[None])
        right_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("right_sole_link")[None])
        parent_pose = get_mid_sole_link_pose(left_sole, right_sole)
    else:
        parent_pose = jaxlie.SE3.identity((1,))
    welding_object_pose = welding_object_pose_world
    welding_object = WeldObject(**welding_object_config)
    return welding_object, welding_object_pose_world, welding_object_pose, parent_pose

def make_target_poses(welding_path):
    return jnp.asarray(welding_path)

# compose_transforms 함수 추가

def compose_transforms(x1: float, y1: float, z1: float, yaw1: float,
                      x2: float, y2: float, z2: float, yaw2: float):
    so3_1 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw1)
    T1 = jaxlie.SE3.from_rotation_and_translation(so3_1, jnp.array([x1, y1, z1]))
    so3_2 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw2)
    T2 = jaxlie.SE3.from_rotation_and_translation(so3_2, jnp.array([x2, y2, z2]))
    T_result = T1 @ T2
    translation = T_result.translation()
    rpy = T_result.rotation().as_rpy_radians()
    return float(translation[0]), float(translation[1]), float(translation[2]), float(rpy[2])

# sample_one_mid_sole_pose_from_search_space 함수 수정

def sample_one_mid_sole_pose_from_search_space(target_x, target_y, target_z, target_yaw, search_space):
    relative_x = np.random.uniform(*search_space['x_range'])
    relative_y = np.random.uniform(*search_space['y_range'])
    relative_z = search_space.get('z_height', 0.0)
    relative_yaw = np.random.uniform(*search_space['angle_range'])
    mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = compose_transforms(
        target_x, target_y, target_z, target_yaw,
        relative_x, relative_y, relative_z, relative_yaw
    )
    return mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw

def create_robot_base_transform(mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw, modified_urdf):
    T_world_mid_sole = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(0.0, 0.0, mid_sole_yaw),
        jnp.array([mid_sole_x, mid_sole_y, mid_sole_z])
    )
    left_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("left_sole_link"))
    right_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("right_sole_link"))
    T_robot_base_mid_sole = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp((left_sole.rotation().log() + right_sole.rotation().log()) / 2),
        translation=(left_sole.translation() + right_sole.translation()) / 2,
    )
    return T_world_mid_sole @ T_robot_base_mid_sole.inverse()

def solve_eetrack_optimization_with_base_transform(robot, robot_collision, target_poses, weights, T_world_robot_base, safety_margin=0.05, max_iterations=100, collision_pairs=None):
    timesteps = target_poses.shape[0]
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    coll_capsules = robot_collision.coll
    active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
    link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]
    if len(active_idx_i) == 0:
        active_idx_i = jnp.array([0])
        active_idx_j = jnp.array([0])
    def se3_from_pose(pose):
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(pose[3:]), pose[:3]
        )
    @jaxls.Cost.create_factory
    def path_tracking_cost_t(var_values, var_robot_cfg_t, target_pose_t):
        robot_cfg = var_values[var_robot_cfg_t]
        end_effector_link_idx = robot.links.names.index("end_effector")
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_robot_base_arr = fk_poses_arr[end_effector_link_idx]
        T_robot_base_ee = jaxlie.SE3(ee_pose_in_robot_base_arr)
        ee_pose = T_world_robot_base @ T_robot_base_ee
        target_se3 = se3_from_pose(target_pose_t)
        se3_error = target_se3.inverse() @ ee_pose
        error = se3_error.log()
        if error.shape != (6,):
            error = jnp.zeros(6)
        position_error = error[:3]
        orientation_error = error[3:]
        weighted_error = jnp.concatenate([
            position_error * weights["position_tracking"],
            orientation_error * weights["orientation_tracking"],
        ])
        return weighted_error
    @jaxls.Cost.create_factory
    def smoothness_cost_t(var_values, var_robot_cfg_curr, var_robot_cfg_prev):
        curr_cfg = var_values[var_robot_cfg_curr]
        prev_cfg = var_values[var_robot_cfg_prev]
        return (curr_cfg - prev_cfg) * weights["smoothness"]
    @jaxls.Cost.create_factory
    def collision_cost_t(var_values, var_robot_cfg_t):
        robot_cfg = var_values[var_robot_cfg_t]
        return collision_cost_jax(
            robot_cfg, robot, coll_capsules, active_idx_i, active_idx_j,
            safety_margin, weights["collision"], link_indices_for_collision
        )
    costs = []
    for t in range(timesteps):
        costs.append(path_tracking_cost_t(var_joints[t], target_poses[t]))
        costs.append(pk.costs.limit_cost(robot, var_joints[t], weights["joint_limits"]))
        costs.append(collision_cost_t(var_joints[t]))
    for t in range(timesteps - 1):
        costs.append(smoothness_cost_t(var_joints[t+1], var_joints[t]))
    termination_config = TerminationConfig(max_iterations=max_iterations)
    solution = (
        jaxls.LeastSquaresProblem(costs, [var_joints])
        .analyze()
        .solve(termination=termination_config)
    )
    solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])
    return solved_joints

def convert_collision_pairs_to_indices(collision_pairs, robot_collision):
    link_names = robot_collision.link_names
    link_name_to_idx = {name: i for i, name in enumerate(link_names)}
    active_idx_i = []
    active_idx_j = []
    for pair in collision_pairs:
        if pair[0] in link_name_to_idx and pair[1] in link_name_to_idx:
            active_idx_i.append(link_name_to_idx[pair[0]])
            active_idx_j.append(link_name_to_idx[pair[1]])
    return jnp.array(active_idx_i), jnp.array(active_idx_j)

def main():
    # Load configs
    asset_dir = Path(__file__).parent / "eetrack"
    task_config_file = asset_dir / "welding_task_config.yaml"
    with open(task_config_file, "r") as f:
        task_config = yaml.safe_load(f)
    config, _ = load_config()
    robot, modified_urdf, robot_collision = load_robot(config)
    # Get search space and batch config
    search_space = task_config['search_space']
    batch_cfg = task_config.get('batch_config', {})
    num_samples = batch_cfg.get('default_num_samples', 10)
    output_dir = Path(batch_cfg.get('output_dir', 'welding_task_results'))
    output_dir.mkdir(exist_ok=True)
    # For each task
    for task_name, task in task_config['tasks'].items():
        print(f"\n=== Task: {task_name} ===")
        welding_obj_pose = task['welding_object']
        target_x = welding_obj_pose['x']
        target_y = welding_obj_pose['y']
        target_z = welding_obj_pose['z']
        target_yaw = welding_obj_pose['yaw']
        # Create welding object
        welding_object, welding_object_pose_world, welding_object_pose, _ = get_welding_object_and_pose(
            config, modified_urdf, target_x, target_y, target_yaw, target_z)
        # Get welding path
        pose_params = jnp.expand_dims(welding_object_pose.parameters(), axis=0)
        welding_path_se3 = welding_object.get_welding_path(jaxlie.SE3(pose_params))
        welding_path_pos = welding_path_se3.translation()
        welding_path_xyzw = jnp.roll(welding_path_se3.rotation().wxyz, shift=-1, axis=-1)
        welding_path = jnp.concatenate([welding_path_pos, welding_path_xyzw], axis=-1)[0]
        target_poses = make_target_poses(welding_path)
        weights = config['weights']
        collision_cfg = config.get('collision', {})
        collision_pairs = config.get('collision_pairs', [])
        max_iterations = config.get('optimization', {}).get('max_iterations', 30)
        safety_margin = collision_cfg.get('safety_margin', 0.01)
        # Batch sampling
        results = []
        for i in range(num_samples):
            mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = sample_one_mid_sole_pose_from_search_space(
                target_x, target_y, target_z, target_yaw, search_space)
            T_world_robot_base = create_robot_base_transform(mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw, modified_urdf)
            try:
                joints = solve_eetrack_optimization_with_base_transform(
                    robot, robot_collision, target_poses, weights, T_world_robot_base,
                    safety_margin=safety_margin, max_iterations=max_iterations, collision_pairs=collision_pairs
                )
                # 간단한 에러 분석 (최대 position/orientation error)
                end_effector_link_idx = robot.links.names.index("end_effector")
                fk_poses_arr = robot.forward_kinematics(cfg=joints[-1])
                ee_pose_in_robot_base_arr = fk_poses_arr[end_effector_link_idx]
                T_robot_base_ee = jaxlie.SE3(ee_pose_in_robot_base_arr)
                ee_pose = T_world_robot_base @ T_robot_base_ee
                target_se3 = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3.from_quaternion_xyzw(target_poses[-1, 3:]),
                    target_poses[-1, :3]
                )
                error = (target_se3.inverse() @ ee_pose).log()
                position_error = float(jnp.linalg.norm(error[:3]))
                orientation_error = float(jnp.linalg.norm(error[3:]))
                results.append({
                    'mid_sole_x': float(mid_sole_x),
                    'mid_sole_y': float(mid_sole_y),
                    'mid_sole_z': float(mid_sole_z),
                    'mid_sole_yaw': float(mid_sole_yaw),
                    'position_error': position_error,
                    'orientation_error': orientation_error,
                    'success': True
                })
                print(f"  [{i+1}/{num_samples}] Success: pos_err={position_error:.4f}, ori_err={orientation_error:.4f}")
            except Exception as e:
                print(f"  [{i+1}/{num_samples}] Failed: {e}")
                results.append({
                    'mid_sole_x': float(mid_sole_x),
                    'mid_sole_y': float(mid_sole_y),
                    'mid_sole_z': float(mid_sole_z),
                    'mid_sole_yaw': float(mid_sole_yaw),
                    'success': False,
                    'error': str(e)
                })
        # Save results
        output_path = output_dir / f"{task_name}_mid_sole_batch_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {output_path}")

if __name__ == "__main__":
    main() 