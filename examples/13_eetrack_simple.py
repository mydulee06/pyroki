import numpy as np
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
import yaml
import yourdfpy
from pathlib import Path
from typing import Tuple, TypedDict
from eetrack.utils.weld_objects import WeldObject
from jaxls import TerminationConfig, TrustRegionConfig
from pyroki.collision._robot_collision_custom import RobotCollision


class TrackingWeights(TypedDict):
    position_tracking: float
    orientation_tracking: float
    smoothness: float
    joint_limits: float


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
        
        # collision cost 계산
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


def solve_eetrack_optimization(
    robot: pk.Robot,
    robot_collision,
    target_poses: jnp.ndarray,  # (T, 7)
    weights: TrackingWeights,
    safety_margin: float = 0.05,
    max_iterations = 100,
    collision_pairs=None,
) -> Tuple[tuple[jaxlie.SE3, ...], jnp.ndarray]:
    timesteps = target_poses.shape[0]
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    coll_capsules = robot_collision.coll
    
    active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
    
    link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]
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
    )
    
    print(f"Starting optimization with {timesteps} timesteps and max {max_iterations} iterations...")
    solution = (
        jaxls.LeastSquaresProblem(costs, [var_joints])
        .analyze()
        .solve(
            termination = termination_config,
        )
    )
    print("Optimization completed.")
    
    solved_Ts_world_root = tuple([jaxlie.SE3.identity() for _ in range(timesteps)])
    solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])
    return solved_Ts_world_root, solved_joints


def analyze_collision_costs(robot, robot_collision, joints, safety_margin, collision_weight, link_indices_for_collision, collision_pairs, topk=10):
    import numpy as np

    active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
    
    num_pairs = len(active_idx_i)
    max_costs = np.zeros(num_pairs)
    max_dists = np.zeros(num_pairs)
    max_timestep = np.zeros(num_pairs, dtype=int)
    
    for t, robot_cfg in enumerate(joints):
        costs, dists = compute_collision_costs(
            robot, robot_collision.coll, robot_cfg, 
            active_idx_i, active_idx_j, 
            safety_margin, collision_weight, 
            link_indices_for_collision
        )
        costs = np.array(costs)
        dists = np.array(dists)
        update_mask = costs > max_costs
        max_costs[update_mask] = costs[update_mask]
        max_dists[update_mask] = dists[update_mask]
        max_timestep[update_mask] = t
    
    sorted_indices = np.argsort(-max_costs)
    print(f"\n[최대 collision cost 기준 Top {topk} pairs]")
    any_printed = False
    for rank in range(topk):
        i = sorted_indices[rank]
        if max_costs[i] <= 0:
            continue
        link_names = robot_collision.link_names
        pair = (link_names[active_idx_i[i]], link_names[active_idx_j[i]])
        print(f"  {rank+1:2d}: {pair} | max_cost={max_costs[i]:.6f} | dist={max_dists[i]:.6f} | timestep={max_timestep[i]}")
        any_printed = True
    if not any_printed:
        print("  (No active collision cost in any timestep)")


def main():
    print("=== EE Track Optimization (Simplified Version) ===")
    
    # Load configuration and robot
    print("Loading configuration and robot...")
    config, asset_dir = load_config()
    collision_cfg = config['collision']
    safety_margin = collision_cfg.get('safety_margin', 0.01)
    collision_pairs = config.get('collision_pairs', [])

    sampled_x, sampled_y, sampled_yaw, sampled_z = sample_welding_object_pose(config)
    robot, modified_urdf, robot_collision = load_robot(config)
    
    # Generate welding path
    print("Generating welding path...")
    welding_path = get_welding_path(config, asset_dir, modified_urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)
    target_poses = make_target_poses(welding_path)  # (T, 7) jnp.ndarray
    num_timesteps = target_poses.shape[0]
    print(f"Generated welding path with {num_timesteps} timesteps")
    
    # Setup optimization weights
    weights = TrackingWeights(
        position_tracking=config['weights']['position_tracking'],
        orientation_tracking=config['weights']['orientation_tracking'],
        smoothness=config['weights']['smoothness'],
        joint_limits=config['weights']['joint_limits'],
        collision=config['weights'].get('collision', 1.0),
    )
    max_iterations = config.get('optimization', {}).get('max_iterations', 30)
    
    print(f"Optimization weights: {weights}")
    
    # Solve optimization
    print("\n=== Running Optimization ===")
    Ts_world_root, joints = solve_eetrack_optimization(
        robot, robot_collision, target_poses, weights,
        safety_margin=safety_margin,
        max_iterations=max_iterations,
        collision_pairs=collision_pairs
    )
    
    # Analyze results
    print("\n=== Analyzing Results ===")
    max_position_error, max_orientation_error, max_collision_cost = analyze_trajectory(
        robot, joints, target_poses, config, collision_pairs, robot_collision, safety_margin, weights['collision']
    )
    
    position_failed = max_position_error > config['tolerance']['position_error']
    orientation_failed = max_orientation_error > config['tolerance']['orientation_error']
    collision_failed = max_collision_cost > 0.001  # collision cost threshold
    
    print(f"\n=== Error Analysis ===")
    print(f"Max Position Error: {max_position_error:.4f} m (tolerance: {config['tolerance']['position_error']:.4f} m)")
    print(f"Max Orientation Error: {max_orientation_error:.4f} rad (tolerance: {config['tolerance']['orientation_error']:.4f} rad)")
    print(f"Max Collision Cost: {max_collision_cost:.6f} (threshold: 0.001)")
    
    if position_failed or orientation_failed or collision_failed:
        failed_reasons = []
        if position_failed:
            failed_reasons.append("Position")
        if orientation_failed:
            failed_reasons.append("Orientation")
        if collision_failed:
            failed_reasons.append("Collision")
        print(f"❌ FAILED: {' and '.join(failed_reasons)} max error exceeded tolerance")
    else:
        print(f"✅ PASSED: All errors within tolerance")

    # Analyze collision details
    analyze_collision_costs(
        robot, robot_collision, joints,
        safety_margin, weights['collision'],
        [robot.links.names.index(name) for name in robot_collision.link_names],
        collision_pairs,
        topk=10
    )
    
    print("\n=== Optimization Complete ===")
    print(f"Final joint trajectory shape: {joints.shape}")
    print("Run complete - check results above for detailed analysis.")


if __name__ == "__main__":
    main()