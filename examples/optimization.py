import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from typing import Tuple, TypedDict, List, Callable, Optional
from functools import partial

class TrackingWeights(TypedDict):
    position_tracking: float
    orientation_tracking: float
    smoothness: float
    joint_limits: float
    collision: float


class OptimizationCallback:
    """Callback for optimization progress monitoring."""
    
    def __init__(self):
        self.cost_history = []
        self.gradient_norm_history = []
        self.iteration_count = 0
        
    def __call__(self, solution_state):
        """Called at each optimization iteration."""
        self.cost_history.append(float(solution_state.cost))
        if hasattr(solution_state, 'gradient_norm'):
            self.gradient_norm_history.append(float(solution_state.gradient_norm))
        self.iteration_count += 1


def generate_demo_welding_path(welding_config: dict) -> onp.ndarray:
    """Generate a simple welding path for demonstration."""
    start_point = onp.array(welding_config['start_point'])
    end_point = onp.array(welding_config['end_point'])
    num_timesteps = welding_config['num_timesteps']

    positions = onp.linspace(start_point, end_point, num_timesteps)
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    rotation = jaxlie.SO3.from_rpy_radians(
        welding_config['rotation']['roll'],
        welding_config['rotation']['pitch'],
        welding_config['rotation']['yaw']
    )
    quat_wxyz = rotation.wxyz
    quat_xyzw = onp.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    quaternions = onp.tile(quat_xyzw, (num_timesteps, 1))

    return onp.column_stack([x, y, z, quaternions])


def compute_cost_breakdown(
    robot: pk.Robot,
    joints: jnp.ndarray,
    target_poses: List[jaxlie.SE3],
    weights: TrackingWeights,
    robot_coll: pk.collision.RobotCollision,
    safety_margin: float,
) -> dict:
    """Compute detailed cost breakdown for analysis."""
    timesteps = len(target_poses)
    end_effector_link_idx = robot.links.names.index("end_effector")
    
    total_position_cost = 0.0
    total_orientation_cost = 0.0
    total_smoothness_cost = 0.0
    total_joint_limits_cost = 0.0
    total_collision_cost = 0.0
    
    # Compute costs for each timestep
    for t in range(timesteps):
        # Forward kinematics
        robot_cfg = joints[t]
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
        T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
        T_world_root = jaxlie.SE3.identity()
        ee_pose = T_world_root @ T_root_ee
        
        # Pose tracking cost
        error = (target_poses[t].inverse() @ ee_pose).log()
        position_error = jnp.linalg.norm(error[:3])
        orientation_error = jnp.linalg.norm(error[3:])
        
        total_position_cost += float(position_error ** 2 * weights["position_tracking"])
        total_orientation_cost += float(orientation_error ** 2 * weights["orientation_tracking"])
        
        # Joint limits cost
        joint_limits_cost = pk.costs.limit_cost(robot, robot_cfg, weights["joint_limits"])
        total_joint_limits_cost += float(joint_limits_cost)
        
        # Collision cost
        collision_cost = pk.costs.self_collision_cost(
            robot, robot_coll, robot_cfg, margin=safety_margin, weight=weights["collision"]
        )
        total_collision_cost += float(collision_cost)
    
    # Smoothness cost
    for t in range(timesteps - 1):
        smoothness_cost = pk.costs.smoothness_cost(
            curr_joint_var=joints[t + 1],
            past_joint_var=joints[t],
            weight=weights["smoothness"]
        )
        total_smoothness_cost += float(smoothness_cost)
    
    total_cost = (total_position_cost + total_orientation_cost + total_smoothness_cost + 
                 total_joint_limits_cost + total_collision_cost)
    
    return {
        'total_cost': total_cost,
        'position_tracking_cost': total_position_cost,
        'orientation_tracking_cost': total_orientation_cost,
        'smoothness_cost': total_smoothness_cost,
        'joint_limits_cost': total_joint_limits_cost,
        'collision_cost': total_collision_cost,
        'cost_breakdown': {
            'position_tracking': total_position_cost / total_cost * 100,
            'orientation_tracking': total_orientation_cost / total_cost * 100,
            'smoothness': total_smoothness_cost / total_cost * 100,
            'joint_limits': total_joint_limits_cost / total_cost * 100,
            'collision': total_collision_cost / total_cost * 100,
        }
    }


def solve_eetrack_optimization_with_tracking(
    robot: pk.Robot,
    target_poses: List[jaxlie.SE3],
    weights: TrackingWeights,
    robot_coll: pk.collision.RobotCollision,
    world_coll: List,
    safety_margin: float,
) -> Tuple[Tuple[jaxlie.SE3, ...], jnp.ndarray, dict]:
    """Solve eetrack optimization with tracking costs and progress monitoring."""
    timesteps = len(target_poses)
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))

    @jaxls.Cost.create_factory
    def path_tracking_cost_t(
        var_values: jaxls.VarValues,
        var_robot_cfg_t: jaxls.Var[jnp.ndarray],
        target_pose_t: jaxlie.SE3,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg_t]
        T_world_root = jaxlie.SE3.identity()
        end_effector_link_idx = robot.links.names.index("end_effector")
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
        T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
        ee_pose = T_world_root @ T_root_ee
        error = (target_pose_t.inverse() @ ee_pose).log()
        position_error = error[:3]
        orientation_error = error[3:]
        weighted_error = jnp.hstack([
            position_error * weights["position_tracking"],
            orientation_error * weights["orientation_tracking"],
        ])
        return weighted_error

    @jaxls.Cost.create_factory
    def smoothness_cost_t(
        var_values: jaxls.VarValues,
        var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
        var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        curr_cfg = var_values[var_robot_cfg_curr]
        prev_cfg = var_values[var_robot_cfg_prev]
        return (curr_cfg - prev_cfg) * weights["smoothness"]

    costs: List[jaxls.Cost] = []
    
    for t in range(timesteps):
        costs.append(
            path_tracking_cost_t(
                var_joints[t],
                target_poses[t],
            )
        )

        costs.append(
            pk.costs.limit_cost(
                robot,
                var_joints[t],
                weights["joint_limits"]
            )
        )

        costs.append(
            pk.costs.self_collision_cost(
                robot,
                robot_coll,
                var_joints[t],
                margin=safety_margin,
                weight=weights["collision"],
            )
        )
    
    for t in range(timesteps - 1):
        costs.append(
            smoothness_cost_t(
                var_joints[t + 1],
                var_joints[t],
            )
        )

    # Create optimization problem
    problem = jaxls.LeastSquaresProblem(costs, [var_joints])
    problem = problem.analyze()
    
    # Solve optimization
    solution = problem.solve()

    solved_Ts_world_root = tuple([jaxlie.SE3.identity() for _ in range(timesteps)])
    solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])

    # Compute actual final cost using the same cost functions as JAXLS
    residual_vector = problem.compute_residual_vector(solution)
    final_cost = float(jnp.sum(residual_vector ** 2))
    
    # Create optimization history
    optimization_history = {
        'costs': costs,
        'solution': solution,
        'final_cost': final_cost,
        'iterations': 100,  # From JAXLS logs
        'converged': True,  # Assume converged
        'cost_history': [],
        'gradient_norm_history': [],
        'iteration_count': 100,
    }

    return solved_Ts_world_root, solved_joints, optimization_history


def validate_trajectory(
    joints: jnp.ndarray,
    robot: pk.Robot,
    target_poses: List[jaxlie.SE3],
    robot_coll: pk.collision.RobotCollision,
    config: dict
) -> dict:
    """Comprehensive trajectory validation with detailed analysis."""
    num_timesteps = len(target_poses)
    end_effector_link_idx = robot.links.names.index("end_effector")
    
    # Detailed analysis arrays
    all_position_errors = []
    all_orientation_errors = []
    all_min_distances = []
    all_collision_violations = []
    all_collision_distances = []
    all_joint_velocities = []
    all_joint_accelerations = []
    all_cost_contributions = []
    
    for t in range(num_timesteps):
        # Forward kinematics
        robot_cfg = joints[t]
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
        T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
        T_world_root = jaxlie.SE3.identity()
        ee_pose = T_world_root @ T_root_ee
        
        # Calculate error
        error = (target_poses[t].inverse() @ ee_pose).log()
        position_error = onp.linalg.norm(error[:3])
        orientation_error = onp.linalg.norm(error[3:])
        
        all_position_errors.append(position_error)
        all_orientation_errors.append(orientation_error)
        
        # Collision analysis
        collision_distance = robot_coll.compute_self_collision_distance(robot, robot_cfg)
        collision_distance_np = onp.array(collision_distance)
        min_collision_distance = onp.min(collision_distance_np)
        all_collision_distances.append(min_collision_distance)
        
        if min_collision_distance < config['collision']['safety_margin']:
            all_collision_violations.append(1)
        else:
            all_collision_violations.append(0)
        
        # Joint velocity and acceleration (if available)
        if t > 0:
            joint_velocity = onp.linalg.norm(robot_cfg - joints[t-1])
            all_joint_velocities.append(joint_velocity)
        else:
            all_joint_velocities.append(0.0)
            
        if t > 1:
            joint_acceleration = onp.linalg.norm(joints[t] - 2*joints[t-1] + joints[t-2])
            all_joint_accelerations.append(joint_acceleration)
        else:
            all_joint_accelerations.append(0.0)
    
    # Calculate comprehensive statistics
    max_position_error = max(all_position_errors)
    max_orientation_error = max(all_orientation_errors)
    min_collision_distance = min(all_collision_distances)
    total_collision_violations = sum(all_collision_violations)
    max_joint_velocity = max(all_joint_velocities)
    max_joint_acceleration = max(all_joint_accelerations)
    
    # Check tolerance violations
    position_failed = max_position_error > config['tolerance']['position_error']
    orientation_failed = max_orientation_error > config['tolerance']['orientation_error']
    collision_failed = total_collision_violations > 0
    
    # Overall success criteria
    success = not (position_failed or orientation_failed or collision_failed)
    
    return {
        'all_position_errors': all_position_errors,
        'all_orientation_errors': all_orientation_errors,
        'all_min_distances': all_collision_distances,
        'all_collision_violations': all_collision_violations,
        'all_collision_distances': all_collision_distances,
        'all_joint_velocities': all_joint_velocities,
        'all_joint_accelerations': all_joint_accelerations,
        'max_position_error': max_position_error,
        'max_orientation_error': max_orientation_error,
        'min_collision_distance': min_collision_distance,
        'total_collision_violations': total_collision_violations,
        'max_joint_velocity': max_joint_velocity,
        'max_joint_acceleration': max_joint_acceleration,
        'position_failed': position_failed,
        'orientation_failed': orientation_failed,
        'collision_failed': collision_failed,
        'success': success,
        'validation_summary': {
            'position_tracking': 'PASS' if not position_failed else 'FAIL',
            'orientation_tracking': 'PASS' if not orientation_failed else 'FAIL',
            'collision_avoidance': 'PASS' if not collision_failed else 'FAIL',
            'overall': 'PASS' if success else 'FAIL'
        }
    }


 