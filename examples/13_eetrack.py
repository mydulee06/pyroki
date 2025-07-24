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
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy
import trimesh
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

def analyze_trajectory(robot, joints, target_poses, config, self_collision_pairs=None, robot_collision=None, world_collision=None, world_collision_pairs=None, safety_margin=None, collision_weight=None):
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
        if robot_collision is not None and safety_margin is not None and collision_weight is not None:
            self_active_idx_i, self_active_idx_j = np.array([]), np.array([])
            if self_collision_pairs is not None:
                self_active_idx_i, self_active_idx_j = convert_collision_pairs_to_indices(self_collision_pairs, robot_collision)
            world_active_idx_i, world_active_idx_j = None, None
            if world_collision is not None and world_collision_pairs:
                world_link_names = [p[0] for p in world_collision_pairs]
                world_object_names = [p[1] for p in world_collision_pairs]
                world_active_idx_i = jnp.array([robot_collision.link_names.index(n) for n in world_link_names])
                world_name_to_idx = {name: i for i, name in enumerate(world_collision.get_object_names())}
                world_active_idx_j = jnp.array([world_name_to_idx[n] for n in world_object_names])
            
            link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]
            costs, _ = compute_collision_costs(
                robot, robot_collision.coll, robot_cfg,
                self_active_idx_i, self_active_idx_j,
                safety_margin, collision_weight,
                link_indices_for_collision,
                world_geom,
                world_pose,
                world_active_idx_i,
                world_active_idx_j
            )
            total_collision_cost = jnp.sum(costs)
            max_collision_cost = jnp.maximum(max_collision_cost, total_collision_cost)
    
    return max_position_error, max_orientation_error, max_collision_cost

def visualize_trajectory(server, urdf_vis, base_frame, Ts_world_root, joints, target_poses_se3, config, robot, position_failed, orientation_failed, robot_collision):
    num_timesteps = len(target_poses_se3)
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    current_error_text = server.gui.add_text("Current Error: ", "Position: 0.0000 m, Orientation: 0.0000 rad")
    status_text = server.gui.add_text("Status: ", "")
    # 최초 상태 동기화
    if position_failed or orientation_failed:
        status_text.value = "❌ FAILED: " + ("Position" if position_failed else "") + (" and " if position_failed and orientation_failed else "") + ("Orientation" if orientation_failed else "") + " max error exceeded tolerance"
    else:
        status_text.value = "✅ PASSED: All errors within tolerance"
    
    # Get collision link indices
    link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]
    
    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            robot_cfg = joints[tstep]
            end_effector_link_idx = robot.links.names.index("end_effector")
            fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
            ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
            ee_se3 = jaxlie.SE3(ee_pose_in_root_arr)
            target_se3 = target_poses_se3[tstep]
            error = (target_se3.inverse() @ ee_se3).log()
            current_position_error = np.linalg.norm(np.array(error[:3]))
            current_orientation_error = np.linalg.norm(np.array(error[3:]))
            current_error_text.value = f"Position: {current_position_error:.4f} m, Orientation: {current_orientation_error:.4f} rad"
            base_frame.wxyz = np.array(Ts_world_root[tstep].wxyz_xyz[:4])
            base_frame.position = np.array(Ts_world_root[tstep].wxyz_xyz[4:])
            urdf_vis.update_cfg(np.array(joints[tstep]))
            server.scene.add_frame(
                "/target_pose",
                axes_length=0.1,
                axes_radius=0.002,
                wxyz=target_se3.rotation().wxyz,
                position=target_se3.translation(),
            )
            
            # Update collision capsules
            fk_results_collision = fk_poses_arr[jnp.array(link_indices_for_collision)]
            coll_world = robot_collision.coll.transform(jaxlie.SE3(fk_results_collision))
            
            # Update each collision capsule
            for i, link_name in enumerate(robot_collision.link_names):
                capsule = jax.tree.map(lambda x: x[i], coll_world)
                capsule_mesh = capsule.to_trimesh()
                
                # Create wireframe by extracting edges and creating thin cylinders
                edges = capsule_mesh.edges_unique
                vertices = capsule_mesh.vertices
                
                # Create thin cylinders for each edge to simulate wireframe
                edge_meshes = []
                for edge in edges:
                    v1, v2 = vertices[edge[0]], vertices[edge[1]]
                    edge_length = np.linalg.norm(v2 - v1)
                    if edge_length > 0.001:  # Only create edge if length is significant
                        # Create thin cylinder for this edge
                        edge_cylinder = trimesh.creation.cylinder(
                            radius=0.001,  # Very thin radius
                            height=edge_length,
                            sections=6
                        )
                        
                        # Position and orient the cylinder
                        center = (v1 + v2) / 2
                        direction = v2 - v1
                        direction_normalized = direction / np.linalg.norm(direction)
                        
                        # Create rotation matrix to align cylinder with edge
                        z_axis = np.array([0, 0, 1])
                        if np.allclose(direction_normalized, z_axis):
                            rotation_matrix = np.eye(3)
                        else:
                            # Find rotation to align z-axis with edge direction
                            rotation_axis = np.cross(z_axis, direction_normalized)
                            if np.linalg.norm(rotation_axis) > 1e-6:
                                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                                angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
                                # Create rotation matrix using Rodrigues' formula
                                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                             [rotation_axis[2], 0, -rotation_axis[0]],
                                             [-rotation_axis[1], rotation_axis[0], 0]])
                                rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                            else:
                                rotation_matrix = np.eye(3)
                        
                        # Apply transform
                        transform_matrix = np.eye(4)
                        transform_matrix[:3, :3] = rotation_matrix
                        transform_matrix[:3, 3] = center
                        edge_cylinder.apply_transform(transform_matrix)
                        
                        edge_meshes.append(edge_cylinder)
                
                # Combine all edge meshes
                if edge_meshes:
                    wireframe_mesh = trimesh.util.concatenate(edge_meshes)
                else:
                    # Fallback: use original mesh
                    wireframe_mesh = capsule_mesh
                
                # Update the mesh in viser scene with red color
                wireframe_mesh.visual.face_colors = [255, 0, 0, 255]  # Red color
                server.scene.add_mesh_trimesh(
                    f"collision_capsule_{link_name}",
                    wireframe_mesh,
                )
        time.sleep(config['visualization']['sleep_time'])

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

def compute_collision_costs(
    robot,
    robot_coll,
    robot_cfg,
    self_active_idx_i, self_active_idx_j,
    safety_margin,
    collision_weight,
    link_indices_for_collision,
    world_robot_collision: RobotCollision = None, 
    world_pose: jaxlie.SE3 = None,
    world_active_idx_i=None,
    world_active_idx_j=None,
):
    Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg=robot_cfg)
    Ts_link_world_se3 = jaxlie.SE3(Ts_link_world_wxyz_xyz)
    coll_robot_world = robot_coll.coll.transform(
        Ts_link_world_se3[jnp.array(link_indices_for_collision)]
    )

    from pyroki.collision._collision import pairwise_collide #

    dist_matrix = pairwise_collide(coll_robot_world, coll_robot_world)
    self_dists = dist_matrix[self_active_idx_i, self_active_idx_j]
    self_costs = jnp.maximum(0, safety_margin - self_dists) * collision_weight

    if world_robot_collision is not None and world_pose is not None:
        coll_world_obj = world_robot_collision.coll.transform(world_pose)
        
        world_dist_matrix = pairwise_collide(coll_robot_world, coll_world_obj)
        world_dists = world_dist_matrix[world_active_idx_i, world_active_idx_j]
        world_costs = jnp.maximum(0, safety_margin - world_dists) * collision_weight

        return jnp.concatenate([self_costs, world_costs]), jnp.concatenate(
            [self_dists, world_dists]
        )

    return self_costs, self_dists

@jax.jit
def collision_cost_jax(
    robot_cfg,
    robot,
    coll_capsules,
    self_active_idx_i,
    self_active_idx_j,
    safety_margin,
    collision_weight,
    link_indices_for_collision,
    world_robot_collision: RobotCollision = None, 
    world_pose: jaxlie.SE3 = None,
    world_active_idx_i=None,
    world_active_idx_j=None
):
    costs, _ = compute_collision_costs(robot, coll_capsules, robot_cfg, self_active_idx_i, self_active_idx_j, safety_margin, collision_weight, link_indices_for_collision, world_geom, world_pose, world_active_idx_i, world_active_idx_j)
    return jnp.array([jnp.sum(costs)])

def solve_eetrack_optimization(
    robot: pk.Robot,
    robot_collision,
    target_poses: jnp.ndarray,  # (T, 7)
    weights: TrackingWeights,
    world_robot_collision: RobotCollision = None, 
    world_pose: jaxlie.SE3 = None,
    safety_margin: float = 0.05,
    max_iterations = 100,
    self_collision_pairs=None,
    world_collision_pairs=None,
) -> Tuple[tuple[jaxlie.SE3, ...], jnp.ndarray]:
    timesteps = target_poses.shape[0]
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    coll_capsules = robot_collision.coll

    self_active_idx_i, self_active_idx_j = convert_collision_pairs_to_indices(self_collision_pairs, robot_collision)
    world_active_idx_i, world_active_idx_j = None, None
    if world_geom is not None and world_pose is not None and world_collision_pairs:
        world_link_names = [p[0] for p in world_collision_pairs]
        world_active_idx_i = jnp.array([robot_collision.link_names.index(n) for n in world_link_names])
        world_active_idx_j = jnp.zeros_like(world_active_idx_i)
    
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
            self_active_idx_i,
            self_active_idx_j,
            safety_margin,
            weights["collision"],
            link_indices_for_collision,
            world_geom,
            world_pose,
            world_active_idx_i,
            world_active_idx_j
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
    config, asset_dir = load_config()
    collision_cfg = config['collision']
    safety_margin = collision_cfg.get('safety_margin', 0.01)
    self_collision_pairs = config.get('collision_pairs', [])

    sampled_x, sampled_y, sampled_yaw, sampled_z = sample_welding_object_pose(config)
    robot, modified_urdf, robot_collision = load_robot(config)
    
    # Get welding object, pose, and parent pose (single source of truth)
    welding_object, welding_object_pose, parent_pose = (None, None, None)
    if config["welding_path_from_object"]:
        welding_object, welding_object_pose, parent_pose = get_welding_object_and_pose(
            config, modified_urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)

    world_robot_collision = None
    world_collision_pairs = []
    if welding_object is not None:
        welding_object_name = "welding_target"
        world_coll_geom = Convex(convex=[welding_object.trimesh])
        world_robot_collision = RobotCollision(
            coll=world_coll_geom,
            link_names=[welding_object_name],
        )
        world_collision_pairs = [
            ("wrist_2_link", welding_object_name),
            ("wrist_3_link", welding_object_name),
            ("tcp", welding_object_name),
            ("end_effector", welding_object_name),
        ]
    welding_path = get_welding_path(config, asset_dir, modified_urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)
    target_poses = make_target_poses(welding_path)  # (T, 7) jnp.ndarray
    target_poses_se3 = make_target_poses_se3(welding_path)  # SE3 object list (for visualization/analysis)
    num_timesteps = target_poses.shape[0]
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, modified_urdf, root_node_name="/base")
    weights = TrackingWeights(
        position_tracking=config['weights']['position_tracking'],
        orientation_tracking=config['weights']['orientation_tracking'],
        smoothness=config['weights']['smoothness'],
        joint_limits=config['weights']['joint_limits'],
        collision=config['weights'].get('collision', 1.0),
    )
    max_iterations = config.get('optimization', {}).get('max_iterations', 30)
    # Add welding_object, welding_object_pose, object_parent to viser.scene (as in old version)
    if welding_object is not None and welding_object_pose is not None:
        server.scene.add_mesh_trimesh("welding_object", welding_object.trimesh.apply_transform(welding_object_pose.as_matrix()[0]))
        server.scene.add_frame(
            "welding_object_pose",
            axes_length=0.1,
            axes_radius=0.002,
            wxyz=welding_object_pose.rotation().wxyz[0],
            position=welding_object_pose.translation()[0],
        )
        server.scene.add_frame(
            "/object_parent",
            axes_length=0.1,
            axes_radius=0.002,
            wxyz=parent_pose.rotation().wxyz[0],
            position=parent_pose.translation()[0],
        )
    
    # Add collision capsules as wireframes
    def add_collision_capsules():
        # Get initial robot configuration (use sit terminal state)
        sit_terminal_states = np.load(config['robot']['sit_terminal_states_path'])
        idx = np.abs(sit_terminal_states["target_height"] - config['robot']['sit_target_height']).argmin()
        initial_joint_pos = sit_terminal_states["joint_state"][idx, 0]
        
        # Map to robot's joint configuration
        urdf_obj = yourdfpy.URDF.load(config['robot']['urdf_path'])
        lab2yourdf = [np.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in urdf_obj.actuated_joint_names]
        initial_joints = initial_joint_pos[lab2yourdf]
        
        # Ensure correct shape
        if len(initial_joints) != robot.joints.num_actuated_joints:
            print(f"Shape mismatch! Using zeros instead.")
            initial_joints = np.zeros(robot.joints.num_actuated_joints)
        
        # Get FK results for initial configuration
        fk_results = robot.forward_kinematics(cfg=initial_joints)
        
        # Extract only the FK results for collision links
        link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]
        fk_results_collision = fk_results[jnp.array(link_indices_for_collision)]
        
        # Get collision capsules in world frame
        coll_world = robot_collision.coll.transform(jaxlie.SE3(fk_results_collision))
        
        # Add each capsule as wireframe
        for i, link_name in enumerate(robot_collision.link_names):
            # Extract single capsule using tree_map
            capsule = jax.tree.map(lambda x: x[i], coll_world)
            capsule_mesh = capsule.to_trimesh()
            
            # Create wireframe by extracting edges and creating thin cylinders
            edges = capsule_mesh.edges_unique
            vertices = capsule_mesh.vertices
            
            # Create thin cylinders for each edge to simulate wireframe
            edge_meshes = []
            for edge in edges:
                v1, v2 = vertices[edge[0]], vertices[edge[1]]
                edge_length = np.linalg.norm(v2 - v1)
                if edge_length > 0.001:  # Only create edge if length is significant
                    # Create thin cylinder for this edge
                    edge_cylinder = trimesh.creation.cylinder(
                        radius=0.001,  # Very thin radius
                        height=edge_length,
                        sections=6
                    )
                    
                    # Position and orient the cylinder
                    center = (v1 + v2) / 2
                    direction = v2 - v1
                    direction_normalized = direction / np.linalg.norm(direction)
                    
                    # Create rotation matrix to align cylinder with edge
                    z_axis = np.array([0, 0, 1])
                    if np.allclose(direction_normalized, z_axis):
                        rotation_matrix = np.eye(3)
                    else:
                        # Find rotation to align z-axis with edge direction
                        rotation_axis = np.cross(z_axis, direction_normalized)
                        if np.linalg.norm(rotation_axis) > 1e-6:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                            angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
                            # Create rotation matrix using Rodrigues' formula
                            K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                         [rotation_axis[2], 0, -rotation_axis[0]],
                                         [-rotation_axis[1], rotation_axis[0], 0]])
                            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                        else:
                            rotation_matrix = np.eye(3)
                    
                    # Apply transform
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = rotation_matrix
                    transform_matrix[:3, 3] = center
                    edge_cylinder.apply_transform(transform_matrix)
                    
                    edge_meshes.append(edge_cylinder)
            
            # Combine all edge meshes
            if edge_meshes:
                wireframe_mesh = trimesh.util.concatenate(edge_meshes)
            else:
                # Fallback: use original mesh
                wireframe_mesh = capsule_mesh
            
                        # Add to viser scene with red color
                # Set red color for the mesh
                wireframe_mesh.visual.face_colors = [255, 0, 0, 255]  # Red color
                server.scene.add_mesh_trimesh(
                    f"collision_capsule_{link_name}",
                    wireframe_mesh,
                )
    
    add_collision_capsules()

    Ts_world_root, joints = solve_eetrack_optimization(
        robot, robot_collision, target_poses, weights,
        world_robot_collision=world_robot_collision,
        world_pose=welding_object_pose[0],
        safety_margin=safety_margin,
        max_iterations=max_iterations,
        self_collision_pairs=self_collision_pairs,
        world_collision_pairs=world_collision_pairs,
    )
    # Error analysis based on SE3 object list (as in old version)
    max_position_error, max_orientation_error, max_collision_cost = analyze_trajectory(
        robot, joints, target_poses, config,
        self_collision_pairs=self_collision_pairs,
        robot_collision=robot_collision,
        world_robot_collision=world_robot_collision,
        world_pose=welding_object_pose[0],
        safety_margin=safety_margin,
        world_collision_pairs=world_collision_pairs,
        collision_weight=weights['collision']
    )
    position_failed = max_position_error > config['tolerance']['position_error']
    orientation_failed = max_orientation_error > config['tolerance']['orientation_error']
    collision_failed = max_collision_cost > 0.001  # collision cost threshold
    print(f"=== Error Analysis ===")
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

    analyze_collision_costs(
        robot, robot_collision, joints,
        safety_margin, weights['collision'],
        [robot.links.names.index(name) for name in robot_collision.link_names],
        self_collision_pairs,
        world_collision_pairs=world_collision_pairs,
        topk=10
    )
    # Use target_poses_se3 in visualization loop
    visualize_trajectory(server, urdf_vis, base_frame, Ts_world_root, joints, target_poses_se3, config, robot, position_failed, orientation_failed, robot_collision)

if __name__ == "__main__":
    main()
