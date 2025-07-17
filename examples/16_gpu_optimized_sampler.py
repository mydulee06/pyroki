import time
from pathlib import Path
from typing import Tuple, TypedDict, List, Dict, Any
from functools import partial
import math
import yaml
from io import StringIO
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle
import json
import os
import logging

# Set logging level to reduce optimization output
logging.getLogger().setLevel(logging.WARNING)

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy
from jax import vmap, jit, pmap
from jax.random import PRNGKey, split, uniform

# GPU optimization settings
jax.config.update('jax_enable_x64', True)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

class TrackingWeights(TypedDict):
    position_tracking: float
    orientation_tracking: float
    smoothness: float
    joint_limits: float
    collision: float


class WeldingPath(TypedDict):
    start_point: jnp.ndarray
    end_point: jnp.ndarray
    rotation: Dict[str, float]
    num_timesteps: int
    success: bool
    max_position_error: float
    max_orientation_error: float
    total_cost: float
    collision_violations: int
    min_collision_distance: float
    optimization_time: float


def setup_gpu_environment():
    """Setup GPU environment."""
    print("Setting up GPU environment...")
    
    # Check GPU availability
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            print(f"Found {len(gpu_devices)} GPU device(s): {gpu_devices}")
            # GPU available
            print("GPU will be used for computation")
        else:
            print("No GPU devices found, using CPU")
    except Exception as e:
        print(f"GPU setup error: {e}, using CPU")


def load_robot_and_config(config_path: Path) -> Tuple[pk.Robot, pk.collision.RobotCollision, dict]:
    """Load robot and configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and modify URDF
    urdf_obj = yourdfpy.URDF.load(config['robot']['urdf_path'])
    movable_joints_set = set(config['robot']['movable_joints'])
    
    xml_tree = urdf_obj.write_xml()
    for joint in xml_tree.findall('.//joint[@type="revolute"]'):
        if joint.get("name") not in movable_joints_set:
            joint.set("type", "fixed")
    
    modified_urdf = yourdfpy.URDF.load(StringIO(ET.tostring(xml_tree.getroot(), encoding="unicode")))
    robot = pk.Robot.from_urdf(modified_urdf)
    
    # Collision detection setup
    collision_config = config.get('collision', {})
    ignore_adjacent = collision_config.get('ignore_adjacent_links', True)
    ignore_pairs = collision_config.get('ignore_pairs', [])
    only_movable_links = collision_config.get('only_movable_links', True)
    exclude_links = collision_config.get('exclude_links', [])
    
    # Convert ignore_pairs to tuple
    user_ignore_pairs = tuple(tuple(pair) for pair in ignore_pairs)
    
    # Consider only movable links
    if only_movable_links:
        all_link_names = list(urdf_obj.link_map.keys())
        movable_link_names = set()
        for joint_name in movable_joints_set:
            if joint_name in urdf_obj.joint_map:
                joint = urdf_obj.joint_map[joint_name]
                movable_link_names.add(joint.parent)
                movable_link_names.add(joint.child)
        
        fixed_link_names = set(all_link_names) - movable_link_names
        
        # Add excluded links to ignore_pairs
        for excluded_link in exclude_links:
            if excluded_link in all_link_names:
                for other_link in all_link_names:
                    if excluded_link != other_link:
                        pair = sorted([excluded_link, other_link])
                        if pair not in ignore_pairs:
                            ignore_pairs.append(pair)
        
        # Ignore collisions only between fixed links
        remaining_fixed_links = fixed_link_names - set(exclude_links)
        for fixed_link1 in remaining_fixed_links:
            for fixed_link2 in remaining_fixed_links:
                if fixed_link1 != fixed_link2:
                    pair = sorted([fixed_link1, fixed_link2])
                    if pair not in ignore_pairs:
                        ignore_pairs.append(pair)
        
        user_ignore_pairs = tuple(tuple(pair) for pair in ignore_pairs)
    
    robot_coll = pk.collision.RobotCollision.from_urdf(
        urdf_obj,
        ignore_immediate_adjacents=ignore_adjacent,
        user_ignore_pairs=user_ignore_pairs
    )
    
    print(f"RobotCollision created with {robot_coll.num_links} links and {len(robot_coll.active_idx_i)} active collision pairs")
    return robot, robot_coll, config


def generate_random_welding_paths(
    key: jnp.ndarray,
    num_paths: int,
    search_config: dict,
    rotation_config: dict
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate random welding paths."""
    
    # Define search space
    x_range = search_config['x_range']
    y_range = search_config['y_range']
    angle_range = search_config['angle_range']
    line_length = search_config['line_length']
    z_height = search_config['z_height']
    
    # Split random keys
    keys = split(key, num_paths)
    
    def generate_single_path(key):
        # Generate random start point and angle
        x_start = uniform(key, (), minval=x_range[0], maxval=x_range[1])
        y_start = uniform(key, (), minval=y_range[0], maxval=y_range[1])
        angle = uniform(key, (), minval=angle_range[0], maxval=angle_range[1])
        
        # Calculate start and end points
        start_point = jnp.array([x_start, y_start, z_height])
        end_point = jnp.array([
            x_start + line_length * jnp.cos(angle),
            y_start + line_length * jnp.sin(angle),
            z_height
        ])
        
        return start_point, end_point, angle
    
    # Vectorized path generation
    start_points, end_points, angles = vmap(generate_single_path)(keys)
    
    return start_points, end_points, angles


def generate_target_poses_batch(
    start_points: jnp.ndarray,
    end_points: jnp.ndarray,
    rotation_config: dict,
    num_timesteps: int
) -> jaxlie.SE3:
    """Generate target poses in batch."""
    
    # Position interpolation
    t_values = jnp.linspace(0.0, 1.0, num_timesteps)
    positions = start_points[:, None, :] + t_values[None, :, None] * (end_points[:, None, :] - start_points[:, None, :])
    
    # Rotation generation
    rotation = jaxlie.SO3.from_rpy_radians(
        rotation_config['roll'],
        rotation_config['pitch'],
        rotation_config['yaw']
    )
    rotations = jax.vmap(lambda _: rotation)(jnp.arange(num_timesteps))
    
    # Add batch dimension
    batch_rotations = jax.vmap(lambda _: rotations)(jnp.arange(start_points.shape[0]))
    
    # Create SE3 poses
    poses = jaxlie.SE3.from_rotation_and_translation(batch_rotations, positions)
    
    return poses


@partial(jit, static_argnames=("safety_margin",))
def solve_optimization_batch(
    robot: pk.Robot,
    target_poses: jaxlie.SE3,
    weights: TrackingWeights,
    robot_coll: pk.collision.RobotCollision,
    safety_margin: float,
) -> jnp.ndarray:
    """Perform optimization in batch."""
    batch_size = target_poses.translation().shape[0]
    timesteps = target_poses.translation().shape[1]
    end_effector_link_idx = robot.links.names.index("end_effector")
    
    # Decompose target_poses into individual timesteps
    target_positions = target_poses.translation()  # (batch, timesteps, 3)
    target_rotations = target_poses.rotation()     # (batch, timesteps, 4)
    
    # Convert SO3 objects to quaternion arrays
    target_quaternions = target_rotations.wxyz  # (batch, timesteps, 4)
    
    def solve_single_path(target_positions_seq, target_quaternions_seq):
        var_joints = robot.joint_var_cls(jnp.arange(timesteps))
        costs = []
        
        for t in range(timesteps):
            # Create SE3 object for each timestep
            target_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_quaternion_xyzw(target_quaternions_seq[t]),
                target_positions_seq[t]
            )
            
            # Pose cost
            costs.append(
                pk.costs.pose_cost(
                    robot=robot,
                    joint_var=var_joints[t],
                    target_pose=target_pose,
                    target_link_index=jnp.array(end_effector_link_idx, dtype=jnp.int32),
                    pos_weight=weights["position_tracking"],
                    ori_weight=weights["orientation_tracking"],
                )
            )
            
            # Limit cost
            costs.append(
                pk.costs.limit_cost(
                    robot=robot,
                    joint_var=var_joints[t],
                    weight=weights["joint_limits"]
                )
            )
            
            # Collision cost
            costs.append(
                pk.costs.self_collision_cost(
                    robot=robot,
                    robot_coll=robot_coll,
                    joint_var=var_joints[t],
                    margin=safety_margin,
                    weight=weights["collision"],
                )
            )
        
        # Smoothness cost
        for t in range(timesteps - 1):
            costs.append(
                pk.costs.smoothness_cost(
                    curr_joint_var=var_joints[t + 1],
                    past_joint_var=var_joints[t],
                    weight=weights["smoothness"]
                )
            )
        
        # Execute optimization
        solution = (
            jaxls.LeastSquaresProblem(costs, [var_joints])
            .analyze()
            .solve()
        )
        
        solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])
        return solved_joints
    
    # Batch processing - pass position and quaternion separately
    solved_joints = vmap(solve_single_path)(target_positions, target_quaternions)
    
    return solved_joints


@partial(jit, static_argnames=("position_error_tolerance", "orientation_error_tolerance"))
def evaluate_paths_batch(
    robot: pk.Robot,
    joints: jnp.ndarray,
    target_poses: jaxlie.SE3,
    robot_coll: pk.collision.RobotCollision,
    position_error_tolerance: float,
    orientation_error_tolerance: float,
    safety_margin: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate paths in batch."""
    # Decompose target_poses into individual timesteps
    target_positions = target_poses.translation()  # (batch, timesteps, 3)
    target_rotations = target_poses.rotation()     # (batch, timesteps, 4)
    
    # Convert SO3 objects to quaternion arrays
    target_quaternions = target_rotations.wxyz  # (batch, timesteps, 4)
    timesteps = target_positions.shape[1]
    end_effector_link_idx = robot.links.names.index("end_effector")
    
    def evaluate_single_path(joint_sequence, target_positions_seq, target_quaternions_seq):
        # Forward kinematics
        fk_poses = vmap(lambda cfg: robot.forward_kinematics(cfg=cfg))(joint_sequence)
        ee_poses = jaxlie.SE3(fk_poses[:, end_effector_link_idx])
        
        # Create target SE3 object for each timestep
        def create_target_pose(t):
            return jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_quaternion_xyzw(target_quaternions_seq[t]),
                target_positions_seq[t]
            )
        
        target_poses_seq = vmap(create_target_pose)(jnp.arange(timesteps))
        
        # Error calculation
        def compute_error(ee_pose, target_pose):
            return (target_pose.inverse() @ ee_pose).log()
        
        errors = vmap(compute_error)(ee_poses, target_poses_seq)
        position_errors = jnp.linalg.norm(errors[:, :3], axis=1)
        orientation_errors = jnp.linalg.norm(errors[:, 3:], axis=1)
        
        # Maximum error
        max_position_error = jnp.max(position_errors)
        max_orientation_error = jnp.max(orientation_errors)
        
        # Collision analysis
        collision_distances = vmap(lambda cfg: robot_coll.compute_self_collision_distance(robot, cfg))(joint_sequence)
        min_collision_distance = jnp.min(collision_distances)
        collision_violations = jnp.sum(collision_distances < safety_margin)
        
        # Success criteria (Error tolerance + No collision violations)
        position_failed = max_position_error > position_error_tolerance
        orientation_failed = max_orientation_error > orientation_error_tolerance
        error_failed = jnp.logical_or(position_failed, orientation_failed)
        
        # Fail if there are collision violations
        collision_failed = collision_violations > 0        
        # Both conditions must be satisfied for success
        success = jnp.logical_not(jnp.logical_or(error_failed, collision_failed))
        
        return success, max_position_error, max_orientation_error, min_collision_distance, collision_violations
    
    # Batch processing - pass position and quaternion separately
    results = vmap(evaluate_single_path)(joints, target_positions, target_quaternions)
    successes, max_pos_errors, max_ori_errors, min_coll_distances, coll_violations = results
    
    return successes, max_pos_errors, max_ori_errors, min_coll_distances, coll_violations


def save_successful_paths(
    successful_paths: List[WeldingPath],
    output_dir: Path,
    config: dict,
    metadata: dict
):
    """Save successful paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save successful paths to files
    timestamp = time.strftime("%Y%m%d_%H%M%S")    
    # Save as JSON format
    json_data = {
        'metadata': metadata,
        'config': config,
        'successful_paths': [
            {
                'start_point': path['start_point'].tolist(),
                'end_point': path['end_point'].tolist(),
                'rotation': path['rotation'],
                'num_timesteps': path['num_timesteps'],
                'max_position_error': float(path['max_position_error']),
                'max_orientation_error': float(path['max_orientation_error']),
                'total_cost': float(path['total_cost']),
                'collision_violations': int(path['collision_violations']),
                'min_collision_distance': float(path['min_collision_distance']),
                'optimization_time': float(path['optimization_time'])
            }
            for path in successful_paths
        ]
    }
    
    json_path = output_dir / f"successful_paths_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    # Also save as Pickle format (preserve JAX arrays)
    pickle_path = output_dir / f"successful_paths_{timestamp}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(successful_paths, f)
    
    print(f"Saved {len(successful_paths)} successful paths to:")
    print(f"  JSON: {json_path}")
    print(f"  Pickle: {pickle_path}")


def main():
    # Set up GPU environment
    setup_gpu_environment()
    
    # Load configuration
    config_path = Path(__file__).parent / "gpu_sampling_config.yaml"
    print("Loading robot and configuration...")
    robot, robot_coll, config = load_robot_and_config(config_path)
    
    # Sampling configuration
    sampling_config = {
        'num_paths': config['sampling']['num_paths'],
        'batch_size': config['sampling']['batch_size'],
        'num_timesteps': config['optimization']['num_timesteps'],
        'safety_margin': config['collision']['safety_margin'],
        'weights': config['weights'],
        'tolerance': config['tolerance'],
        'search_space': config['search_space'],
        'rotation': config['welding_orientation']
    }
    
    print(f"Sampling {sampling_config['num_paths']} welding paths...")
    print(f"Batch size: {sampling_config['batch_size']}")
    print(f"Using device: {jax.devices()}")
    
    # Initialize random key
    key = PRNGKey(42)
    
    successful_paths = []
    total_processed = 0
    
    # Process in batches
    num_batches = (sampling_config['num_paths'] + sampling_config['batch_size'] - 1) // sampling_config['batch_size']
    
    start_time = time.time()
    
    with tqdm(total=sampling_config['num_paths'], desc="Processing welding paths") as pbar:
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            
            # Calculate current batch size
            current_batch_size = min(sampling_config['batch_size'], 
                                   sampling_config['num_paths'] - batch_idx * sampling_config['batch_size'])
            
            # Update random key
            key, batch_key = split(key)
            
            # Generate random paths
            start_points, end_points, angles = generate_random_welding_paths(
                batch_key, current_batch_size, sampling_config['search_space'], sampling_config['rotation']
            )
            
            # Generate target poses
            target_poses = generate_target_poses_batch(
                start_points, end_points, sampling_config['rotation'], sampling_config['num_timesteps']
            )
            
            # Execute optimization
            solved_joints = solve_optimization_batch(
                robot, target_poses, sampling_config['weights'], robot_coll, sampling_config['safety_margin']
            )
            
            # Evaluate paths
            successes, max_pos_errors, max_ori_errors, min_coll_distances, coll_violations = evaluate_paths_batch(
                robot, solved_joints, target_poses, robot_coll, 
                sampling_config['tolerance']['position_error'], 
                sampling_config['tolerance']['orientation_error'],
                sampling_config['safety_margin']
            )
            
            batch_time = time.time() - batch_start_time
            
            # Collect successful paths
            for i in range(current_batch_size):
                if successes[i]:
                    path = WeldingPath(
                        start_point=start_points[i],
                        end_point=end_points[i],
                        rotation=sampling_config['rotation'],
                        num_timesteps=sampling_config['num_timesteps'],
                        success=True,
                        max_position_error=float(max_pos_errors[i]),
                        max_orientation_error=float(max_ori_errors[i]),
                        total_cost=0.0,  # TODO: Add actual cost calculation
                        collision_violations=int(coll_violations[i]),
                        min_collision_distance=float(min_coll_distances[i]),
                        optimization_time=batch_time / current_batch_size
                    )
                    successful_paths.append(path)
            
            total_processed += current_batch_size
            pbar.update(current_batch_size)
            pbar.set_postfix({
                'Success Rate': f"{len(successful_paths)/total_processed*100:.1f}%",
                'Successful': len(successful_paths),
                'Batch Time': f"{batch_time:.2f}s"
            })
    
    end_time = time.time()
    
    print(f"\n=== Sampling Complete ===")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Total paths processed: {total_processed}")
    print(f"Successful paths: {len(successful_paths)}")
    print(f"Success rate: {len(successful_paths)/total_processed*100:.2f}%")
    print(f"Average time per path: {(end_time - start_time)/total_processed:.4f} seconds")
    
    # Save successful paths
    if successful_paths:
        output_dir = Path(config['output']['output_dir'])
        metadata = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_processed': total_processed,
            'successful_count': len(successful_paths),
            'success_rate': len(successful_paths)/total_processed*100,
            'processing_time': end_time - start_time,
            'average_time_per_path': (end_time - start_time)/total_processed,
            'device': str(jax.devices())
        }
        
        save_successful_paths(successful_paths, output_dir, config, metadata)
        
        # Print statistics of successful paths
        print(f"\n=== Successful Paths Statistics ===")
        max_pos_errors = [path['max_position_error'] for path in successful_paths]
        max_ori_errors = [path['max_orientation_error'] for path in successful_paths]
        min_coll_distances = [path['min_collision_distance'] for path in successful_paths]
        opt_times = [path['optimization_time'] for path in successful_paths]
        
        print(f"Position errors - Min: {min(max_pos_errors):.6f}, Max: {max(max_pos_errors):.6f}, Avg: {sum(max_pos_errors)/len(max_pos_errors):.6f}")
        print(f"Orientation errors - Min: {min(max_ori_errors):.6f}, Max: {max(max_ori_errors):.6f}, Avg: {sum(max_ori_errors)/len(max_ori_errors):.6f}")
        print(f"Min collision distances - Min: {min(min_coll_distances):.6f}, Max: {max(min_coll_distances):.6f}, Avg: {sum(min_coll_distances)/len(min_coll_distances):.6f}")
        print(f"Optimization times - Min: {min(opt_times):.4f}s, Max: {max(opt_times):.4f}s, Avg: {sum(opt_times)/len(opt_times):.4f}s")
    else:
        print("No successful paths found. Consider adjusting parameters.")


if __name__ == "__main__":
    main() 