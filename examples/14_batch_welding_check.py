import time
from pathlib import Path
from typing import Tuple, TypedDict
from functools import partial
import math
import yaml
from io import StringIO
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy

# --- Data Structures ---

class TrackingWeights(TypedDict):
    position_tracking: float
    orientation_tracking: float
    smoothness: float
    joint_limits: float
    rest_pose: float

# --- JAX-native Core Functions ---

def generate_target_poses(start_point: jnp.ndarray, end_point: jnp.ndarray, rotation_rpy: dict, num_timesteps: int) -> jaxlie.SE3:
    """Generates a sequence of SE(3) target poses for a single welding line."""
    positions = jnp.linspace(start_point, end_point, num_timesteps)
    rotation = jaxlie.SO3.from_rpy_radians(
        rotation_rpy['roll'], rotation_rpy['pitch'], rotation_rpy['yaw']
    )
    # Tile rotation for all timesteps
    rotations = jax.vmap(lambda _: rotation)(jnp.arange(num_timesteps))
    return jaxlie.SE3.from_rotation_and_translation(rotations, positions)

@partial(jax.jit, static_argnames=("robot", "weights", "actuated_joint_names", "movable_joint_names"))
def solve_eetrack_optimization(
    robot: pk.Robot,
    target_poses: jaxlie.SE3,
    weights: TrackingWeights,
    actuated_joint_names: tuple[str, ...],
    movable_joint_names: tuple[str, ...],
) -> Tuple[jnp.int32, jnp.ndarray]:
    """Solve the EETrack optimization problem for a single trajectory."""
    timesteps = target_poses.shape[0]
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))

    # --- Cost definitions ---
    @jaxls.Cost.create_factory
    def path_tracking_cost_t(
        var_values: jaxls.VarValues,
        var_robot_cfg_t: jaxls.Var[jnp.ndarray],
        target_pose_t: jaxlie.SE3,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg_t]
        end_effector_link_idx = robot.links.names.index("end_effector")
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
        ee_pose = jaxlie.SE3(ee_pose_in_root_arr)
        error = (target_pose_t.inverse() @ ee_pose).log()
        return jnp.hstack([
            error[:3] * weights["position_tracking"],
            error[3:] * weights["orientation_tracking"],
        ])

    @jaxls.Cost.create_factory
    def smoothness_cost_t(
        var_values: jaxls.VarValues,
        var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
        var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        return (var_values[var_robot_cfg_curr] - var_values[var_robot_cfg_prev]) * weights["smoothness"]

    # --- Build costs list ---
    costs: list[jaxls.Cost] = []
    rest_weights = jnp.full(var_joints.default_factory().shape, weights["rest_pose"])
    for i, joint_name in enumerate(actuated_joint_names):
        if joint_name not in movable_joint_names:
            rest_weights = rest_weights.at[i].set(weights["rest_pose"] * 100.0)

    for t in range(timesteps):
        costs.append(path_tracking_cost_t(var_joints[t], target_poses[t]))
        costs.append(pk.costs.limit_cost(robot, var_joints[t], weights["joint_limits"]))
        costs.append(pk.costs.rest_cost(var_joints[t], var_joints.default_factory(), rest_weights))

    for t in range(timesteps - 1):
        costs.append(smoothness_cost_t(var_joints[t + 1], var_joints[t]))

    # --- Solve ---
    solution = jaxls.LeastSquaresProblem(costs, [var_joints]).analyze().solve()
    solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])
    return solution.status, solved_joints

@partial(jax.jit, static_argnames=("robot",))
def check_feasibility_jax(robot: pk.Robot, joints: jnp.ndarray, target_poses: jaxlie.SE3, tolerance_config: dict) -> Tuple[jnp.bool_, jnp.ndarray, jnp.ndarray]:
    """JAX-native feasibility check for a single trajectory."""
    end_effector_link_idx = robot.links.names.index("end_effector")

    def fk_for_one_timestep(robot_cfg):
        fk_poses = robot.forward_kinematics(cfg=robot_cfg)
        return jaxlie.SE3(fk_poses[end_effector_link_idx])

    ee_poses = jax.vmap(fk_for_one_timestep)(joints)
    errors = (target_poses.inverse() @ ee_poses).log()

    position_errors = jnp.linalg.norm(errors[:, :3], axis=1)
    orientation_errors = jnp.linalg.norm(errors[:, 3:], axis=1)

    max_position_error = jnp.max(position_errors)
    max_orientation_error = jnp.max(orientation_errors)

    position_failed = max_position_error > tolerance_config['position_error']
    orientation_failed = max_orientation_error > tolerance_config['orientation_error']

    is_feasible = jnp.logical_not(jnp.logical_or(position_failed, orientation_failed))
    return is_feasible, max_position_error, max_orientation_error

def main():
    # --- Configuration and Setup ---
    asset_dir = Path(__file__).parent / "retarget_helpers" / "eetrack"
    with open(asset_dir / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    urdf_obj = yourdfpy.URDF.load(config['robot']['urdf_path'])
    movable_joints_set = set(config['robot']['movable_joints'])
    xml_tree = urdf_obj.write_xml()
    for joint in xml_tree.findall('.//joint[@type="revolute"]'):
        if joint.get("name") not in movable_joints_set:
            joint.set("type", "fixed")
    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(StringIO(ET.tostring(xml_tree.getroot(), encoding="unicode"))))

    # --- Generate Candidate Lines ---
    search_config = config['search_space']
    x_params, y_params, angle_params = search_config['x_range'], search_config['y_range'], search_config['angle_range']
    x_range = onp.arange(x_params[0], x_params[1], x_params[2])
    y_range = onp.arange(y_params[0], y_params[1], y_params[2])
    angle_range = onp.arange(angle_params[0], angle_params[1], angle_params[2])

    candidate_lines = []
    for x_start in x_range:
        for y_start in y_range:
            for angle in angle_range:
                start_point = onp.array([x_start, y_start, search_config['z_height']])
                end_point = onp.array([
                    x_start + search_config['line_length'] * onp.cos(angle),
                    y_start + search_config['line_length'] * onp.sin(angle),
                    search_config['z_height']
                ])
                candidate_lines.append({'start': start_point, 'end': end_point})

    if not candidate_lines:
        print("No candidate lines generated. Check your search_space config.")
        return

    print(f"Generated {len(candidate_lines)} candidate welding lines to check.")
    
    candidate_starts_jnp = jnp.array([line['start'] for line in candidate_lines])
    candidate_ends_jnp = jnp.array([line['end'] for line in candidate_lines])

    # --- Define Batched Processing Function ---
    weights = config['weights']
    actuated_joint_names = tuple(robot.joints.actuated_names)
    movable_joint_names = tuple(config['robot']['movable_joints'])
    num_timesteps = config['search_space']['num_timesteps']
    rotation_rpy = config['welding_path']['rotation']
    tolerance_config = config['tolerance']

    def check_one_line(start_point, end_point):
        """Processes a single line: generates poses, solves IK, and checks feasibility."""
        target_poses = generate_target_poses(start_point, end_point, rotation_rpy, num_timesteps)
        
        status, solved_joints = solve_eetrack_optimization(
            robot, target_poses, weights, actuated_joint_names, movable_joint_names
        )

        def feasible_branch(_):
            return check_feasibility_jax(robot, solved_joints, target_poses, tolerance_config)
        
        def non_feasible_branch(_):
            return jnp.array(False), jnp.array(jnp.inf), jnp.array(jnp.inf)

        return jax.lax.cond(
            status == jaxls.SolveStatus.SOLVED,
            feasible_branch,
            non_feasible_branch,
            None
        )

    # --- Compile and Run ---
    print("Compiling batch processing function with JAX... (this may take a minute)")
    jitted_batch_checker = jax.jit(jax.vmap(check_one_line))

    start_time = time.time()
    all_feasible, all_pos_errs, all_orient_errs = jitted_batch_checker(candidate_starts_jnp, candidate_ends_jnp)
    all_feasible.block_until_ready() # Ensure computation is finished for accurate timing
    end_time = time.time()
    
    # --- Report Results ---
    print("\n\n--- Batch Processing Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")

    feasible_mask = onp.array(all_feasible)
    feasible_indices = onp.where(feasible_mask)[0]
    
    print(f"Found {len(feasible_indices)} feasible welding lines out of {len(candidate_lines)} candidates.")
    
    if feasible_indices.any():
        print("Feasible lines:")
        for i in feasible_indices:
            line = candidate_lines[i]
            pos_err = all_pos_errs[i]
            orient_err = all_orient_errs[i]
            print(f"  Start: {onp.round(line['start'], 2)}, End: {onp.round(line['end'], 2)} | "
                  f"Pos Err: {pos_err:.4f}m, Orient Err: {orient_err:.4f}rad")
    else:
        print("No feasible lines found in the defined search space.")

if __name__ == "__main__":
    main() 