

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, TypedDict
import sys
import os
import threading
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import functions and classes
from eetrack.utils.weld_objects import WeldObject
import pyroki as pk
import yourdfpy
from pyroki.collision._robot_collision_custom import RobotCollision
import viser
from viser.extras import ViserUrdf
import trimesh
import jaxls
from jaxls import TerminationConfig


class TrackingWeights(TypedDict):
    position_tracking: float
    orientation_tracking: float
    smoothness: float
    joint_limits: float
    collision: float


class State:
    """Shared state between main thread and visualization thread"""
    def __init__(self, num_timesteps):
        self.joints = jnp.zeros((num_timesteps, 18))  # Default joint configuration
        self.Ts_world_root = [jaxlie.SE3.identity() for _ in range(num_timesteps)]
        self.target_poses_se3 = None
        self.max_position_error = 0.0
        self.max_orientation_error = 0.0
        self.max_collision_cost = 0.0
        self.position_failed = False
        self.orientation_failed = False
        self.collision_failed = False


def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    """Calculate mid point between left and right sole links"""
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + right_sole_link_pose.rotation().log()) / 2
        ),
        translation=(left_sole_link_pose.translation() + right_sole_link_pose.translation()) / 2,
    )


def load_config():
    """Load configuration from YAML file"""
    asset_dir = Path(__file__).parent / "eetrack"
    config_file = asset_dir / "config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config, asset_dir


def load_robot(config):
    """Load robot from URDF and create collision model"""
    urdf_path = config['robot']['urdf_path']
    urdf_obj = yourdfpy.URDF.load(urdf_path)
    
    # Set joint positions from sit terminal states
    sit_terminal_states = np.load(config['robot']['sit_terminal_states_path'])
    idx = np.abs(sit_terminal_states["target_height"] - config['robot']['sit_target_height']).argmin()
    joint_pos = sit_terminal_states["joint_state"][idx, 0]
    lab2yourdf = [np.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in urdf_obj.actuated_joint_names]
    urdf_obj.update_cfg(joint_pos[lab2yourdf])
    
    # Fix non-movable joints
    for joint in urdf_obj.robot.joints:
        if joint.name in urdf_obj.actuated_joint_names and joint.name not in config['robot']['movable_joints']:
            joint.type = "fixed"
            joint.origin = urdf_obj.get_transform(joint.child, joint.parent)
    
    modified_urdf = yourdfpy.URDF(urdf_obj.robot, mesh_dir=Path(urdf_path).parent)

    # Create collision model
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
    """Create welding object and its pose"""
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
    
    # For trajectory optimization, use parent-relative pose
    welding_object_pose = welding_object_pose_world
    
    welding_object = WeldObject(**welding_object_config)
    return welding_object, welding_object_pose_world, welding_object_pose, parent_pose


def make_target_poses(welding_path):
    """Convert welding path to target poses array"""
    return jnp.asarray(welding_path)


def make_target_poses_se3(welding_path):
    """Convert welding path to SE3 object list for visualization"""
    return [
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(np.array(welding_path[i, 3:])),
            np.array(welding_path[i, :3])
        )
        for i in range(welding_path.shape[0])
    ]


def convert_collision_pairs_to_indices(collision_pairs, robot_collision):
    """Convert collision pairs to indices for collision computation"""
    link_names = robot_collision.link_names
    link_name_to_idx = {name: i for i, name in enumerate(link_names)}
    active_idx_i = []
    active_idx_j = []
    for pair in collision_pairs:
        if pair[0] in link_name_to_idx and pair[1] in link_name_to_idx:
            active_idx_i.append(link_name_to_idx[pair[0]])
            active_idx_j.append(link_name_to_idx[pair[1]])
    return jnp.array(active_idx_i), jnp.array(active_idx_j)


def compute_collision_costs(robot, coll_capsules, robot_cfg, active_idx_i, active_idx_j, safety_margin, collision_weight, link_indices_for_collision):
    """Compute collision costs for given robot configuration"""
    Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg=robot_cfg)
    Ts_link_world_wxyz_xyz = Ts_link_world_wxyz_xyz[jnp.array(link_indices_for_collision)]
    coll_world = coll_capsules.transform(jaxlie.SE3(Ts_link_world_wxyz_xyz))
    from pyroki.collision._collision import pairwise_collide
    dist_matrix = pairwise_collide(coll_world, coll_world)
    dists = dist_matrix[active_idx_i, active_idx_j]
    costs = jnp.maximum(0, safety_margin - dists) * collision_weight
    return costs, dists


def collision_cost_jax(robot_cfg, robot, coll_capsules, active_idx_i, active_idx_j, safety_margin, collision_weight, link_indices_for_collision):
    """JAX-compatible collision cost function"""
    if len(active_idx_i) == 0 or len(active_idx_j) == 0:
        return jnp.zeros(1)
    try:
        costs, _ = compute_collision_costs(robot, coll_capsules, robot_cfg, active_idx_i, active_idx_j, safety_margin, collision_weight, link_indices_for_collision)
        return jnp.array([jnp.sum(costs)])
    except Exception:
        return jnp.zeros(1)


def solve_eetrack_optimization_with_base_transform(robot, robot_collision, target_poses, weights, T_world_robot_base, safety_margin=0.05, max_iterations=100, collision_pairs=None):
    """Solve trajectory optimization with robot base transformation"""
    timesteps = target_poses.shape[0]
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    coll_capsules = robot_collision.coll
    
    active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
    link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]
    
    if len(active_idx_i) == 0:
        print("Warning: No active collision pairs found, collision cost will be zero")
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
    
    termination_config = TerminationConfig(max_iterations=max_iterations, early_termination=False)
    solution = (
        jaxls.LeastSquaresProblem(costs, [var_joints])
        .analyze()
        .solve(termination=termination_config)
    )
    solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])
    return solved_joints


def analyze_trajectory_with_base_transform(robot, joints, target_poses, T_world_robot_base, collision_pairs, robot_collision, safety_margin, collision_weight):
    """Analyze trajectory errors and collision costs"""
    num_timesteps = joints.shape[0]
    max_position_error = 0.0
    max_orientation_error = 0.0
    max_collision_cost = 0.0
    
    for t in range(num_timesteps):
        robot_cfg = joints[t]
        end_effector_link_idx = robot.links.names.index("end_effector")
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_robot_base_arr = fk_poses_arr[end_effector_link_idx]
        T_robot_base_ee = jaxlie.SE3(ee_pose_in_robot_base_arr)
        ee_pose = T_world_robot_base @ T_robot_base_ee
        target_se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(target_poses[t, 3:]),
            target_poses[t, :3]
        )
        error = (target_se3.inverse() @ ee_pose).log()
        position_error = jnp.linalg.norm(error[:3])
        orientation_error = jnp.linalg.norm(error[3:])
        max_position_error = jnp.maximum(max_position_error, position_error)
        max_orientation_error = jnp.maximum(max_orientation_error, orientation_error)
        
        if collision_pairs is not None and robot_collision is not None:
            link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]
            active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
            costs, _ = compute_collision_costs(
                robot, robot_collision.coll, robot_cfg,
                active_idx_i, active_idx_j,
                safety_margin, collision_weight,
                link_indices_for_collision
            )
            total_collision_cost = jnp.sum(costs)
            max_collision_cost = jnp.maximum(max_collision_cost, total_collision_cost)
    
    return max_position_error, max_orientation_error, max_collision_cost


def load_inverse_results(json_file: str) -> List[Dict[str, Any]]:
    """Load successful samples from inverse results JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    successful_samples = [sample for sample in data if sample.get('success', False)]
    print(f"Loaded {len(data)} total samples, {len(successful_samples)} successful samples")
    return successful_samples


def compose_transforms(x1: float, y1: float, z1: float, yaw1: float,
                      x2: float, y2: float, z2: float, yaw2: float) -> Tuple[float, float, float, float]:
    """Compose two SE(3) transforms: T_result = T1 @ T2"""
    so3_1 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw1)
    T1 = jaxlie.SE3.from_rotation_and_translation(so3_1, jnp.array([x1, y1, z1]))
    so3_2 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw2)
    T2 = jaxlie.SE3.from_rotation_and_translation(so3_2, jnp.array([x2, y2, z2]))
    T_result = T1 @ T2
    translation = T_result.translation()
    rpy = T_result.rotation().as_rpy_radians()
    return float(translation[0]), float(translation[1]), float(translation[2]), float(rpy[2])


def sample_one_mid_sole_pose(target_welding_object_x: float, 
                            target_welding_object_y: float, 
                            target_welding_object_z: float, 
                            target_welding_object_yaw: float,
                            inverse_results: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """Sample one good mid_sole pose from inverse results"""
    if not inverse_results:
        raise ValueError("No successful samples available")
    
    print(f"🔄 Converting relative poses to world coordinates...")
    print(f"📍 Target welding object (world): x={target_welding_object_x:.3f}, y={target_welding_object_y:.3f}, z={target_welding_object_z:.3f}, yaw={target_welding_object_yaw:.3f}")
    
    successful_samples = [sample for sample in inverse_results if sample['success']]
    if not successful_samples:
        raise ValueError("No successful samples available")
    
    sample = random.choice(successful_samples)

    relative_x = sample['sampled_x']
    relative_y = sample['sampled_y']
    relative_z = sample['sampled_z']
    relative_yaw = sample['sampled_yaw']
    
    print(f"   Using random sample: relative pose x={relative_x:.3f}, y={relative_y:.3f}, z={relative_z:.3f}, yaw={relative_yaw:.3f}")
    
    mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = compose_transforms(
        target_welding_object_x, target_welding_object_y, target_welding_object_z, target_welding_object_yaw,
        relative_x, relative_y, relative_z, relative_yaw
    )
    
    print(f"   -> world coordinates: x={mid_sole_x:.3f}, y={mid_sole_y:.3f}, z={mid_sole_z:.3f}, yaw={mid_sole_yaw:.3f}")
    
    return mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw


def sample_one_mid_sole_pose_from_search_space(target_welding_object_x: float, 
                            target_welding_object_y: float, 
                            target_welding_object_z: float, 
                            target_welding_object_yaw: float,
                            search_space: Dict[str, Any]):
    print(f"🔄 Converting search space to world coordinates...")
    print(f"📍 Target welding object (world): x={target_welding_object_x:.3f}, y={target_welding_object_y:.3f}, z={target_welding_object_z:.3f}, yaw={target_welding_object_yaw:.3f}")
    
    relative_x = np.random.uniform(*search_space['x_range'])
    relative_y = np.random.uniform(*search_space['y_range'])
    relative_z = search_space.get('z_height', 0.0)
    relative_yaw = np.random.uniform(*search_space['angle_range'])
    
    print(f"   Using random sample: relative pose x={relative_x:.3f}, y={relative_y:.3f}, z={relative_z:.3f}, yaw={relative_yaw:.3f}")
    
    mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = compose_transforms(
        target_welding_object_x, target_welding_object_y, target_welding_object_z, target_welding_object_yaw,
        relative_x, relative_y, relative_z, relative_yaw
    )
    
    print(f"   -> world coordinates: x={mid_sole_x:.3f}, y={mid_sole_y:.3f}, z={mid_sole_z:.3f}, yaw={mid_sole_yaw:.3f}")
    
    return mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw


def create_robot_base_transform(mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw, modified_urdf):
    """Create robot base transform from mid_sole pose"""
    T_world_mid_sole = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(0.0, 0.0, mid_sole_yaw),
        jnp.array([mid_sole_x, mid_sole_y, mid_sole_z])
    )
    
    # Get transform from robot base to mid_sole_link from URDF
    left_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("left_sole_link"))
    right_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("right_sole_link"))
    T_robot_base_mid_sole = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp((left_sole.rotation().log() + right_sole.rotation().log()) / 2),
        translation=(left_sole.translation() + right_sole.translation()) / 2,
    )
    return T_world_mid_sole @ T_robot_base_mid_sole.inverse()


def run_optimization(config, robot, modified_urdf, robot_collision, welding_object, welding_object_pose, inverse_results, target_x, target_y, target_z, target_yaw, state, search_space=None, sampling_mode="inverse"):
    """Run optimization and update state"""
    print("🔄 Running optimization...")
    # 샘플링 방식 분기
    if sampling_mode == "inverse":
        mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = sample_one_mid_sole_pose(
            target_x, target_y, target_z, target_yaw, inverse_results
        )
    elif sampling_mode == "search_space":
        if search_space is None:
            raise ValueError("search_space argument is required!")
        mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = sample_one_mid_sole_pose_from_search_space(
            target_x, target_y, target_z, target_yaw, search_space
        )
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
    T_world_robot_base = create_robot_base_transform(mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw, modified_urdf)
    
    # Get welding path
    pose_params = jnp.expand_dims(welding_object_pose.parameters(), axis=0)
    welding_path_se3 = welding_object.get_welding_path(jaxlie.SE3(pose_params))
    welding_path_pos = welding_path_se3.translation()
    welding_path_xyzw = jnp.roll(welding_path_se3.rotation().wxyz, shift=-1, axis=-1)
    welding_path = jnp.concatenate([welding_path_pos, welding_path_xyzw], axis=-1)

    welding_path = welding_path[0]
    
    target_poses = make_target_poses(welding_path)
    target_poses_se3 = make_target_poses_se3(welding_path)

    weights = TrackingWeights(**config['weights'])
    collision_cfg = config.get('collision', {})
    
    # Solve optimization
    joints = solve_eetrack_optimization_with_base_transform(
        robot, robot_collision, target_poses, weights, T_world_robot_base,
        safety_margin=collision_cfg.get('safety_margin', 0.01),
        max_iterations=config.get('optimization', {}).get('max_iterations', 30),
        collision_pairs=config.get('collision_pairs', [])
    )
    
    # Analyze results
    max_position_error, max_orientation_error, max_collision_cost = analyze_trajectory_with_base_transform(
        robot, joints, target_poses, T_world_robot_base, 
        config.get('collision_pairs', []), robot_collision, 
        collision_cfg.get('safety_margin', 0.01), weights['collision']
    )
    
    # Create Ts_world_root for visualization
    Ts_world_root = []
    for t in range(joints.shape[0]):
        Ts_world_root.append(T_world_robot_base)
    
    # Update state
    state.joints = joints
    state.Ts_world_root = Ts_world_root
    state.target_poses_se3 = target_poses_se3
    state.max_position_error = max_position_error
    state.max_orientation_error = max_orientation_error
    state.max_collision_cost = max_collision_cost
    state.position_failed = max_position_error > config['tolerance']['position_error']
    state.orientation_failed = max_orientation_error > config['tolerance']['orientation_error']
    state.collision_failed = max_collision_cost > 0.001
    
    # Print results
    print(f"=== Error Analysis ===")
    print(f"Max Position Error: {max_position_error:.4f} m (tolerance: {config['tolerance']['position_error']:.4f} m)")
    print(f"Max Orientation Error: {max_orientation_error:.4f} rad (tolerance: {config['tolerance']['orientation_error']:.4f} rad)")
    print(f"Max Collision Cost: {max_collision_cost:.6f} (threshold: 0.001)")
    
    if state.position_failed or state.orientation_failed or state.collision_failed:
        failed_reasons = []
        if state.position_failed:
            failed_reasons.append("Position")
        if state.orientation_failed:
            failed_reasons.append("Orientation")
        if state.collision_failed:
            failed_reasons.append("Collision")
        print(f"❌ FAILED: {' and '.join(failed_reasons)} max error exceeded tolerance")
    else:
        print(f"✅ PASSED: All errors within tolerance")
    
    print("✅ Optimization finished. Trajectory updated.")


def animation_loop(server, base_frame, urdf_vis, playing_checkbox, timestep_slider, 
                  current_error_text, status_text, state, config, robot):
    """Animation loop running in background thread"""
    while True:
        with server.atomic():
            if playing_checkbox.value:
                timestep_slider.value = (timestep_slider.value + 1) % len(state.joints)
            
            tstep = timestep_slider.value
            
            if state.target_poses_se3 is not None and tstep < len(state.joints):
                # Calculate current error
                robot_cfg = state.joints[tstep]
                end_effector_link_idx = robot.links.names.index("end_effector")
                fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
                ee_pose_in_robot_base_arr = fk_poses_arr[end_effector_link_idx]
                T_robot_base_ee = jaxlie.SE3(ee_pose_in_robot_base_arr)
                ee_pose = state.Ts_world_root[tstep] @ T_robot_base_ee
                
                target_pose = state.target_poses_se3[tstep]
                error = (target_pose.inverse() @ ee_pose).log()
                current_position_error = np.linalg.norm(error[:3])
                current_orientation_error = np.linalg.norm(error[3:])
                
                # Update error displays
                current_error_text.value = f"Position: {current_position_error:.4f} m, Orientation: {current_orientation_error:.4f} rad"
                
                # Update status based on current error
                if (current_position_error > config['tolerance']['position_error'] or 
                    current_orientation_error > config['tolerance']['orientation_error']):
                    status_text.value = "❌ ERROR: Current error exceeds tolerance"
                elif state.position_failed or state.orientation_failed or state.collision_failed:
                    status_text.value = "❌ FAILED: Max error exceeded tolerance"
                else:
                    status_text.value = "✅ PASSED: All errors within tolerance"
                
                # Update robot visualization
                root_pose = state.Ts_world_root[tstep]
                base_frame.wxyz = np.array(root_pose.rotation().wxyz)
                base_frame.position = np.array(root_pose.translation())
                urdf_vis.update_cfg(np.array(state.joints[tstep]))
                
                # Update target pose visualization
                server.scene.add_frame(
                    "/target_pose",
                    axes_length=0.1,
                    axes_radius=0.002,
                    wxyz=np.array(target_pose.rotation().wxyz),
                    position=np.array(target_pose.translation()),
                )
        
        time.sleep(config['visualization']['sleep_time'])


def main():
    """Main function"""
    # Load task configuration
    asset_dir = Path(__file__).parent / "eetrack"
    config_file = asset_dir / "welding_task_config.yaml"

    with open(config_file, "r") as f:
        task_config = yaml.safe_load(f)

    task_names = list(task_config['tasks'].keys())
    if not task_names:
        print("No tasks defined in welding_task_config.yaml!")
        return
    
    task_name = task_names[0]
    task = task_config['tasks'][task_name]
    search_space = task_config['search_space']
    welding_obj_pose = task['welding_object']
    target_x = welding_obj_pose['x']
    target_y = welding_obj_pose['y']
    target_z = welding_obj_pose['z']
    target_yaw = welding_obj_pose['yaw']

    inverse_results_path = asset_dir / Path(task_config['file_paths']['inverse_results_file'])

    # Load configurations and robot
    config, _ = load_config()
    robot, modified_urdf, robot_collision = load_robot(config)
    
    try:
        inverse_results = load_inverse_results(str(inverse_results_path))
    except Exception as e:
        print(f"❌ Error loading inverse results: {e}")
        return
    
    if not inverse_results:
        print("❌ No successful samples found")
        return

    # Create welding object
    welding_object, welding_object_pose_world, welding_object_pose, _ = get_welding_object_and_pose(
        config, modified_urdf, target_x, target_y, target_yaw, target_z)

    # Initialize state
    num_timesteps = 100  # Default, will be updated after first optimization
    state = State(num_timesteps)

    # Setup viser server and visualization
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, modified_urdf, root_node_name="/base")

    # Add welding object to scene (fixed in world frame for visualization)
    mat = welding_object_pose_world.as_matrix()
    if mat.shape == (1, 4, 4):
        mat = mat[0]  # Remove batch dimension
    server.scene.add_mesh_trimesh("welding_object", welding_object.trimesh.apply_transform(mat))
    server.scene.add_frame(
        "welding_object_pose", axes_length=0.1, axes_radius=0.002,
        wxyz=np.array(welding_object_pose_world.rotation().wxyz),
        position=np.array(welding_object_pose_world.translation()),
    )

    # GUI controls
    playing_checkbox = server.gui.add_checkbox("Playing", True)
    timestep_slider = server.gui.add_slider("Timestep", 0, num_timesteps - 1, 1, 0)
    current_error_text = server.gui.add_text("Current Error", "Position: 0.0000 m, Orientation: 0.0000 rad")
    status_text = server.gui.add_text("Status", "✅ PASSED")

    # Weight tuner
    weights = pk.viewer.WeightTuner(
        server,
        TrackingWeights(
            position_tracking=config['weights']['position_tracking'],
            orientation_tracking=config['weights']['orientation_tracking'],
            smoothness=config['weights']['smoothness'],
            joint_limits=config['weights']['joint_limits'],
            collision=config['weights']['collision'],
        ),
    )

    # 샘플링 방식 선택 드롭다운 추가
    sampling_mode_dropdown = server.gui.add_dropdown("Sampling Mode", ["inverse", "search_space"])

    # Sample & Optimize button
    def on_sample_and_optimize():
        sampling_mode = sampling_mode_dropdown.value
        run_optimization(config, robot, modified_urdf, robot_collision, 
                        welding_object, welding_object_pose, inverse_results, 
                        target_x, target_y, target_z, target_yaw, state,
                        search_space=search_space,
                        sampling_mode=sampling_mode)
        # Update slider range after optimization
        timestep_slider.max = len(state.joints) - 1

    gen_button = server.gui.add_button("Sample & Optimize")
    gen_button.on_click(lambda _: on_sample_and_optimize())

    # Run initial optimization
    on_sample_and_optimize()

    # Start animation loop in background thread
    animation_thread = threading.Thread(
        target=animation_loop,
        args=(server, base_frame, urdf_vis, playing_checkbox, timestep_slider,
              current_error_text, status_text, state, config, robot),
        daemon=True
    )
    animation_thread.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main() 