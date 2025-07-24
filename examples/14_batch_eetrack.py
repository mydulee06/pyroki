import time
from pathlib import Path
from typing import Tuple, TypedDict
import numpy as np
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
import yaml
import yourdfpy
from eetrack.utils.weld_objects import WeldObject
from jaxls import TerminationConfig, TrustRegionConfig
from pyroki.collision._robot_collision_custom import RobotCollision


class TrackingWeights(TypedDict):
    position_tracking: float
    orientation_tracking: float
    smoothness: float
    joint_limits: float
    collision: float

def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + right_sole_link_pose.rotation().log()) / 2
        ),
        translation=(left_sole_link_pose.translation() + right_sole_link_pose.translation()) / 2,
    )

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

def sample_welding_object_pose(config):
    search_space = config.get('search_space', {})
    x_min, x_max = search_space.get('x_range', [-0.3, 0.3])
    y_min, y_max = search_space.get('y_range', [-0.5, -0.1])
    yaw_min, yaw_max = search_space.get('angle_range', [-np.pi/2, np.pi/2])
    z_height = search_space.get('z_height', 0.0)
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    yaw = np.random.uniform(yaw_min, yaw_max)
    print(f"Sampled welding object: x={x:.3f}, y={y:.3f}, z={z_height:.3f}, yaw={yaw:.3f} rad ({np.degrees(yaw):.1f} deg)")
    return x, y, yaw, z_height

def sample_welding_object_pose_batch(config, batch_size):
    search_space = config.get('search_space', {})
    x_min, x_max = search_space.get('x_range', [-0.3, 0.3])
    y_min, y_max = search_space.get('y_range', [-0.5, -0.1])
    yaw_min, yaw_max = search_space.get('angle_range', [-np.pi/2, np.pi/2])
    z_height = search_space.get('z_height', 0.0)
    x = np.random.uniform(x_min, x_max, size=batch_size)
    y = np.random.uniform(y_min, y_max, size=batch_size)
    yaw = np.random.uniform(yaw_min, yaw_max, size=batch_size)
    z = np.full(batch_size, z_height)
    samples = np.stack([x, y, yaw, z], axis=1)
    return samples

def get_welding_object_and_pose(config, modified_urdf, sampled_x=None, sampled_y=None, sampled_yaw=None, sampled_z=None):
    welding_object_config = config["welding_object"].copy()
    welding_object_config.pop('pose', None)
    welding_object_config.pop('yaw', None)
    px = sampled_x or 0.0
    py = sampled_y or 0.0
    pz = sampled_z or 0.0
    yaw = sampled_yaw or 0.0
    so3 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw)
    welding_object_pose = jaxlie.SE3.from_rotation_and_translation(so3, jnp.array([px, py, pz]))
    parent = welding_object_config.pop("parent", None)
    if parent == "mid_sole_link":
        left_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("left_sole_link")[None])
        right_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("right_sole_link")[None])
        parent_pose = get_mid_sole_link_pose(left_sole, right_sole)
    else:
        parent_pose = jaxlie.SE3.identity((1,))
    welding_object_pose = parent_pose @ welding_object_pose
    welding_object = WeldObject(**welding_object_config)
    return welding_object, welding_object_pose, parent_pose

def get_welding_path(config, asset_dir, modified_urdf, sampled_x=None, sampled_y=None, sampled_yaw=None, sampled_z=None):
    if config["welding_path_from_object"]:
        welding_object, welding_object_pose, parent_pose = get_welding_object_and_pose(
            config, modified_urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)
        welding_path_se3 = welding_object.get_welding_path(welding_object_pose)
        welding_path_pos = welding_path_se3.translation()
        welding_path_xyzw = jnp.roll(welding_path_se3.rotation().wxyz, shift=-1, axis=-1)
        welding_path = jnp.concatenate([welding_path_pos, welding_path_xyzw], axis=-1)[0]
    else:
        welding_path_file = asset_dir / "welding_path.npy"
        num_timesteps = config['welding_path']['num_timesteps']
        welding_path = generate_demo_welding_path(config['welding_path'])
        np.save(welding_path_file, welding_path)
    return welding_path

def generate_demo_welding_path(welding_config: dict) -> np.ndarray:
    start_point = np.array(welding_config['start_point'])
    end_point = np.array(welding_config['end_point'])
    num_timesteps = welding_config['num_timesteps']
    positions = np.linspace(start_point, end_point, num_timesteps)
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    rotation = jaxlie.SO3.from_rpy_radians(
        welding_config['rotation']['roll'],
        welding_config['rotation']['pitch'],
        welding_config['rotation']['yaw']
    )
    quat_wxyz = rotation.wxyz
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    quaternions = np.tile(quat_xyzw, (num_timesteps, 1))
    return np.column_stack([x, y, z, quaternions])

def make_target_poses(welding_path):
    # welding_path: (T, 7) ndarray (xyz, xyzw)
    return jnp.asarray(welding_path)

def make_target_poses_se3(welding_path):
    # welding_path: (T, 7) ndarray (xyz, xyzw)
    return [
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(np.array(welding_path[i, 3:])),
            np.array(welding_path[i, :3])
        )
        for i in range(welding_path.shape[0])
    ]

# Vectorized SE3 conversion function
def se3_from_pose(pose):
    # pose: (7,) [x, y, z, x, y, z, w] (xyzw)
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_quaternion_xyzw(pose[3:]), pose[:3]
    )

se3_from_pose_vmap = jax.vmap(se3_from_pose, in_axes=0)

def analyze_trajectory(robot, joints, target_poses, config, collision_pairs=None, robot_collision=None, safety_margin=None, collision_weight=None):
    num_timesteps = joints.shape[0]
    max_position_error = 0.0
    max_orientation_error = 0.0
    max_collision_cost = 0.0
    
    for t in range(num_timesteps):
        robot_cfg = joints[t]
        end_effector_link_idx = robot.links.names.index("end_effector")
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
        T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
        T_world_root = jaxlie.SE3.identity()
        ee_pose = T_world_root @ T_root_ee
        target_se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(target_poses[t, 3:]),
            target_poses[t, :3]
        )
        error = (target_se3.inverse() @ ee_pose).log()
        position_error = jnp.linalg.norm(error[:3])
        orientation_error = jnp.linalg.norm(error[3:])
        max_position_error = jnp.maximum(max_position_error, position_error)
        max_orientation_error = jnp.maximum(max_orientation_error, orientation_error)
        
        if collision_pairs is not None and robot_collision is not None and safety_margin is not None and collision_weight is not None:
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

def make_solve_eetrack_optimization_jitted(robot, robot_collision, weights, max_iterations, collision_pairs, safety_margin):
    active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
    coll_capsules = robot_collision.coll
    link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]

    @jax.jit
    def solve(target_poses):
        timesteps = target_poses.shape[0]
        var_joints = robot.joint_var_cls(jnp.arange(timesteps))

        @jaxls.Cost.create_factory
        def path_tracking_cost_t(
            var_values: jaxls.VarValues,
            var_robot_cfg_t: jaxls.Var[jnp.ndarray],
            target_pose_t: jnp.ndarray,
        ) -> jax.Array:
            robot_cfg = var_values[var_robot_cfg_t]
            T_world_root = jaxlie.SE3.identity()
            end_effector_link_idx = robot.links.names.index("end_effector")
            fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
            ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
            T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
            ee_pose = T_world_root @ T_root_ee
            target_se3 = se3_from_pose(target_pose_t)
            error = (target_se3.inverse() @ ee_pose).log()
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

        @jaxls.Cost.create_factory
        def collision_cost_t(
            var_values: jaxls.VarValues,
            var_robot_cfg_t: jaxls.Var[jnp.ndarray],
        ) -> jax.Array:
            robot_cfg = var_values[var_robot_cfg_t]
            return collision_cost_jax(
                robot_cfg,
                robot,
                coll_capsules,
                active_idx_i,
                active_idx_j,
                safety_margin,
                weights["collision"],
                link_indices_for_collision
            )

        costs = []
        for t in range(timesteps):
            costs.append(path_tracking_cost_t(var_joints[t], target_poses[t]))
            costs.append(pk.costs.limit_cost(robot, var_joints[t], weights["joint_limits"]))
            costs.append(collision_cost_t(var_joints[t]))
        for t in range(timesteps - 1):
            costs.append(smoothness_cost_t(var_joints[t+1], var_joints[t]))
        termination_config = TerminationConfig(
            max_iterations=max_iterations,
            early_termination=False,
        )
        solution = (
            jaxls.LeastSquaresProblem(costs, [var_joints])
            .analyze()
            .solve(
                termination = termination_config,
            )
        )
        solved_Ts_world_root = tuple([jaxlie.SE3.identity() for _ in range(timesteps)])
        solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])
        return solved_Ts_world_root, solved_joints
    return solve


def get_welding_path_batch(config, asset_dir, modified_urdf, sampled_x, sampled_y, sampled_yaw, sampled_z):
    # sampled_x, ...: (B,) ndarray
    B = sampled_x.shape[0]
    welding_paths = []
    for i in range(B):
        welding_path = get_welding_path(
            config, asset_dir, modified_urdf,
            sampled_x[i], sampled_y[i], sampled_yaw[i], sampled_z[i]
        )
        welding_paths.append(welding_path)
    return np.stack(welding_paths, axis=0)  # (B, T, 7)


def process_batch_parallel(config, asset_dir, robot, robot_collision, modified_urdf, weights, max_iterations, samples, solve_fn, collision_pairs, safety_margin, batch_idx=None, num_batches=None):
    # samples: (B, 4)
    B = samples.shape[0]
    sampled_x, sampled_y, sampled_yaw, sampled_z = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    welding_paths = get_welding_path_batch(config, asset_dir, modified_urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)  # (B, T, 7)
    # batch TrajOpt (vmap)
    start_time = time.time()
    Ts_world_root_batch, joints_batch = jax.vmap(solve_fn, in_axes=0)(welding_paths)
    end_time = time.time()
    print(f"Batch TrajOpt completed in {end_time - start_time:.2f} seconds")

    # batch error analysis (vmap)
    start_time = time.time()
    def analyze_fn(joints, target_poses):
        return analyze_trajectory(robot, joints, target_poses, config, collision_pairs, robot_collision, safety_margin, weights['collision'])
    analyze_fn_vmap = jax.vmap(analyze_fn, in_axes=(0, 0))
    max_position_errors, max_orientation_errors, max_collision_costs = analyze_fn_vmap(joints_batch, welding_paths)
    end_time = time.time()
    print(f"Batch Error Analysis completed in {end_time - start_time:.2f} seconds")
    if batch_idx is not None and num_batches is not None:
        print(f"Batch {batch_idx+1}/{num_batches}")

    results = []
    for i in range(B):
        pos_err = float(max_position_errors[i])
        ori_err = float(max_orientation_errors[i])
        coll_cost = float(max_collision_costs[i])
        pos_tol = float(config['tolerance']['position_error'])
        ori_tol = float(config['tolerance']['orientation_error'])
        position_failed = pos_err > pos_tol
        orientation_failed = ori_err > ori_tol
        collision_failed = coll_cost > 0.001  # collision cost threshold
        success = (not position_failed and not orientation_failed and not collision_failed)
        results.append({
            'max_position_error': pos_err,
            'max_orientation_error': ori_err,
            'max_collision_cost': coll_cost,
            'position_failed': bool(position_failed),
            'orientation_failed': bool(orientation_failed),
            'collision_failed': bool(collision_failed),
            'sampled_x': float(sampled_x[i]),
            'sampled_y': float(sampled_y[i]),
            'sampled_yaw': float(sampled_yaw[i]),
            'sampled_z': float(sampled_z[i]),
            'success': success
        })
    return results


def pad_samples(samples, batch_size):
    n = samples.shape[0]
    if n == batch_size:
        return samples, n
    pad = np.zeros((batch_size - n, samples.shape[1]), dtype=samples.dtype)
    padded = np.concatenate([samples, pad], axis=0)
    return padded, n

def save_results(results, filename="batch_eetrack_results.json"):
    import json
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Optimization Results for total {len(results)} samples: {filename}")


def main():
    config, asset_dir = load_config()
    collision_cfg = config['collision']
    safety_margin = collision_cfg.get('safety_margin', 0.05)
    collision_pairs = config.get('collision_pairs', [])
    n_samples = config['search_space'].get('n_samples', 1000)
    batch_size = config['search_space'].get('batch_size', 100)
    robot, modified_urdf, robot_collision = load_robot(config)
    weights = TrackingWeights(
        position_tracking=config['weights']['position_tracking'],
        orientation_tracking=config['weights']['orientation_tracking'],
        smoothness=config['weights']['smoothness'],
        joint_limits=config['weights']['joint_limits'],
        collision=config['weights'].get('collision', 1.0),
    )
    max_iterations = config.get('optimization', {}).get('max_iterations', 30)
    
    # SOLVE function definition with collision
    solve_fn = make_solve_eetrack_optimization_jitted(robot, robot_collision, weights, max_iterations, collision_pairs, safety_margin)

    num_batches = int(np.ceil(n_samples / batch_size))
    all_results = []
    for batch_idx in range(num_batches):
        current_batch_size = batch_size if (batch_idx < num_batches - 1) else (n_samples - batch_idx * batch_size)

        # FUNCTION CALL! (sample_welding_object_pose_batch: (B, 4))
        samples = sample_welding_object_pose_batch(config, current_batch_size)
        samples, valid_n = pad_samples(samples, batch_size)

        start_time = time.time()
        # FUNCTION CALL! (process_batch_parallel: (B, 4)) :: Most Time-Consuming Function
        batch_results = process_batch_parallel(
            config, asset_dir, robot, robot_collision, modified_urdf, weights, max_iterations, samples, solve_fn, collision_pairs, safety_margin,
            batch_idx=batch_idx, num_batches=num_batches
        )
        
        all_results.extend(batch_results[:valid_n])

    save_results(all_results)

if __name__ == "__main__":
    main()
