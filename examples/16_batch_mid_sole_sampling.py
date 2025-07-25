import time
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

def compute_collision_costs(robot, coll_capsules, robot_cfg, active_idx_i, active_idx_j, safety_margin, collision_weight, link_indices_for_collision):
    Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg=robot_cfg)
    Ts_link_world_wxyz_xyz = Ts_link_world_wxyz_xyz[jnp.array(link_indices_for_collision)]
    import jaxlie
    coll_world = coll_capsules.transform(jaxlie.SE3(Ts_link_world_wxyz_xyz))
    from pyroki.collision._collision import pairwise_collide
    dist_matrix = pairwise_collide(coll_world, coll_world)
    dists = dist_matrix[active_idx_i, active_idx_j]
    costs = jnp.maximum(0, safety_margin - dists) * collision_weight
    return costs, dists

@jax.jit
def collision_cost_jax(
    robot_cfg,
    robot,
    coll_capsules,
    active_idx_i,
    active_idx_j,
    safety_margin,
    collision_weight,
    link_indices_for_collision
):
    costs, _ = compute_collision_costs(robot, coll_capsules, robot_cfg, active_idx_i, active_idx_j, safety_margin, collision_weight, link_indices_for_collision)
    return jnp.array([jnp.sum(costs)])

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

def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + right_sole_link_pose.rotation().log()) / 2
        ),
        translation=(left_sole_link_pose.translation() + right_sole_link_pose.translation()) / 2,
    )

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

def sample_mid_sole_poses_batch(target_x, target_y, target_z, target_yaw, search_space, batch_size):
    """Batch sampling of mid_sole poses"""
    relative_x = np.random.uniform(*search_space['x_range'], size=batch_size)
    relative_y = np.random.uniform(*search_space['y_range'], size=batch_size)
    relative_z = np.full(batch_size, search_space.get('z_height', 0.0))
    relative_yaw = np.random.uniform(*search_space['angle_range'], size=batch_size)
    
    # Vectorized transform composition
    mid_sole_poses = []
    for i in range(batch_size):
        mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = compose_transforms(
            target_x, target_y, target_z, target_yaw,
            relative_x[i], relative_y[i], relative_z[i], relative_yaw[i]
        )
        mid_sole_poses.append([mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw])
    
    return np.array(mid_sole_poses)  # (B, 4) [x, y, z, yaw]

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

def create_robot_base_transforms_batch(mid_sole_poses, modified_urdf):
    """Batch creation of robot base transforms"""
    batch_size = mid_sole_poses.shape[0]
    transforms = []
    for i in range(batch_size):
        mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = mid_sole_poses[i]
        T = create_robot_base_transform(mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw, modified_urdf)
        transforms.append(T.parameters())
    return jnp.array(transforms)  # (B, 7) SE3 parameters

def make_solve_eetrack_optimization_with_base_transform_jitted(robot, robot_collision, weights, max_iterations, collision_pairs, safety_margin):
    """Create jitted batch optimization function"""
    active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
    coll_capsules = robot_collision.coll
    link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]
    
    if len(active_idx_i) == 0:
        active_idx_i = jnp.array([0])
        active_idx_j = jnp.array([0])

    @jax.jit
    def solve(target_poses, T_world_robot_base_params):
        T_world_robot_base = jaxlie.SE3(T_world_robot_base_params)
        timesteps = target_poses.shape[0]
        var_joints = robot.joint_var_cls(jnp.arange(timesteps))

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
    
    return solve

def analyze_trajectory_batch(robot, joints_batch, target_poses, T_world_robot_base_batch):
    """Batch analysis of trajectories"""
    def analyze_single(joints, T_world_robot_base_params):
        T_world_robot_base = jaxlie.SE3(T_world_robot_base_params)
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
        position_error = jnp.linalg.norm(error[:3])
        orientation_error = jnp.linalg.norm(error[3:])
        return position_error, orientation_error
    
    analyze_fn_vmap = jax.vmap(analyze_single, in_axes=(0, 0))
    return analyze_fn_vmap(joints_batch, T_world_robot_base_batch)

def process_batch_parallel(robot, robot_collision, target_poses, weights, mid_sole_poses, modified_urdf, 
                         solve_fn, max_iterations, collision_pairs, safety_margin, batch_idx=None, num_batches=None):
    """Process a batch of mid_sole poses in parallel"""
    batch_size = mid_sole_poses.shape[0]
    
    # Create robot base transforms for batch
    T_world_robot_base_batch = create_robot_base_transforms_batch(mid_sole_poses, modified_urdf)
    
    # Batch optimization
    start_time = time.time()
    joints_batch = jax.vmap(solve_fn, in_axes=(None, 0))(target_poses, T_world_robot_base_batch)
    end_time = time.time()
    print(f"Batch TrajOpt completed in {end_time - start_time:.2f} seconds")
    
    # Batch error analysis
    start_time = time.time()
    position_errors, orientation_errors = analyze_trajectory_batch(
        robot, joints_batch, target_poses, T_world_robot_base_batch
    )
    end_time = time.time()
    print(f"Batch Error Analysis completed in {end_time - start_time:.2f} seconds")
    
    if batch_idx is not None and num_batches is not None:
        print(f"Batch {batch_idx+1}/{num_batches}")
    
    # Compile results
    results = []
    for i in range(batch_size):
        pos_err = float(position_errors[i])
        ori_err = float(orientation_errors[i])
        mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = mid_sole_poses[i]
        
        success = pos_err < 0.01 and ori_err < 0.1  # tolerance thresholds
        
        results.append({
            'mid_sole_x': float(mid_sole_x),
            'mid_sole_y': float(mid_sole_y),
            'mid_sole_z': float(mid_sole_z),
            'mid_sole_yaw': float(mid_sole_yaw),
            'position_error': pos_err,
            'orientation_error': ori_err,
            'success': success
        })
    
    return results

def pad_samples(samples, batch_size):
    """Pad samples to match batch size"""
    n = samples.shape[0]
    if n == batch_size:
        return samples, n
    pad = np.zeros((batch_size - n, samples.shape[1]), dtype=samples.dtype)
    padded = np.concatenate([samples, pad], axis=0)
    return padded, n

def calculate_success_rate(results):
    """Calculate success rate from results"""
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    success_rate = successful / total * 100 if total > 0 else 0
    return successful, total, success_rate

def main():
    # Load configs
    asset_dir = Path(__file__).parent / "eetrack"
    task_config_file = asset_dir / "welding_task_config.yaml"
    with open(task_config_file, "r") as f:
        task_config = yaml.safe_load(f)
    
    config, _ = load_config()
    robot, modified_urdf, robot_collision = load_robot(config)
    
    # Get batch configuration
    batch_cfg = task_config.get('batch_config', {})
    n_samples = batch_cfg.get('n_samples', 1000)
    batch_size = batch_cfg.get('batch_size', 100)
    selected_task = batch_cfg.get('selected_task', '')

    
    # Select task to process
    if selected_task and selected_task in task_config['tasks']:
        tasks_to_process = {selected_task: task_config['tasks'][selected_task]}
    else:
        tasks_to_process = task_config['tasks']
        if selected_task:
            print(f"Warning: Task '{selected_task}' not found. Processing all tasks.")
    
    search_space = task_config['search_space']
    weights = config['weights']
    collision_pairs = config.get('collision_pairs', [])
    max_iterations = config.get('optimization', {}).get('max_iterations', 30)
    safety_margin = config.get('collision', {}).get('safety_margin', 0.01)
    
    # Process each task
    overall_results = []
    for task_name, task in tasks_to_process.items():
        print(f"\n=== Task: {task_name} ===")
        
        welding_obj_pose = task['welding_object']
        target_x = welding_obj_pose['x']
        target_y = welding_obj_pose['y']
        target_z = welding_obj_pose['z']
        target_yaw = welding_obj_pose['yaw']
        
        # Create welding object and get target poses
        welding_object, welding_object_pose_world, welding_object_pose, _ = get_welding_object_and_pose(
            config, modified_urdf, target_x, target_y, target_yaw, target_z)
        
        pose_params = jnp.expand_dims(welding_object_pose.parameters(), axis=0)
        welding_path_se3 = welding_object.get_welding_path(jaxlie.SE3(pose_params))
        welding_path_pos = welding_path_se3.translation()
        welding_path_xyzw = jnp.roll(welding_path_se3.rotation().wxyz, shift=-1, axis=-1)
        welding_path = jnp.concatenate([welding_path_pos, welding_path_xyzw], axis=-1)[0]
        target_poses = make_target_poses(welding_path)
        
        # Create jitted solve function
        solve_fn = make_solve_eetrack_optimization_with_base_transform_jitted(
            robot, robot_collision, weights, max_iterations, collision_pairs, safety_margin
        )
        
        # Batch processing
        num_batches = int(np.ceil(n_samples / batch_size))
        all_results = []
        
        for batch_idx in range(num_batches):
            current_batch_size = batch_size if (batch_idx < num_batches - 1) else (n_samples - batch_idx * batch_size)
            
            # Sample mid_sole poses for this batch
            mid_sole_poses = sample_mid_sole_poses_batch(
                target_x, target_y, target_z, target_yaw, search_space, current_batch_size
            )
            mid_sole_poses, valid_n = pad_samples(mid_sole_poses, batch_size)
            
            # Process batch
            try:
                batch_results = process_batch_parallel(
                    robot, robot_collision, target_poses, weights, mid_sole_poses, modified_urdf,
                    solve_fn, max_iterations, collision_pairs, safety_margin,
                    batch_idx=batch_idx, num_batches=num_batches
                )
                all_results.extend(batch_results[:valid_n])
                
            except Exception as e:
                print(f"Batch {batch_idx+1} failed: {e}")
                # Add failed results
                for i in range(valid_n):
                    mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = mid_sole_poses[i]
                    all_results.append({
                        'mid_sole_x': float(mid_sole_x),
                        'mid_sole_y': float(mid_sole_y),
                        'mid_sole_z': float(mid_sole_z),
                        'mid_sole_yaw': float(mid_sole_yaw),
                        'success': False,
                        'error': str(e)
                    })
        
        # Add to overall results
        overall_results.extend(all_results)
        
        # Print task summary
        successful, total, success_rate = calculate_success_rate(all_results)
        print(f"Task {task_name} completed: {successful}/{total} successful optimizations ({success_rate:.1f}%)")
    
    # Print overall summary
    if overall_results:
        print(f"\n{'='*50}")
        overall_successful, overall_total, overall_success_rate = calculate_success_rate(overall_results)
        print(f"OVERALL RESULTS: {overall_successful}/{overall_total} successful optimizations ({overall_success_rate:.1f}%)")
        print(f"{'='*50}")

if __name__ == "__main__":
    main() 