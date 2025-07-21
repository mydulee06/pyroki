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
from eetrack.utils.weld_objects import WeldObject


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
    return pk.Robot.from_urdf(modified_urdf), modified_urdf

def get_welding_path(config, asset_dir, modified_urdf):
    if config["welding_path_from_object"]:
        welding_object_config = config["welding_object"].copy()
        welding_object_pose = jaxlie.SE3(jnp.array(welding_object_config.pop("pose"))[None])
        welding_object_parent = welding_object_config.pop("parent", None)
        if welding_object_parent == "mid_sole_link":
            left_sole_link_pose = jaxlie.SE3.from_matrix(modified_urdf.get_transform("left_sole_link")[None])
            right_sole_link_pose = jaxlie.SE3.from_matrix(modified_urdf.get_transform("right_sole_link")[None])
            parent_pose = get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose)
        else:
            parent_pose = jaxlie.SE3.identity((1,))
        welding_object_pose = parent_pose @ welding_object_pose
        welding_object = WeldObject(**welding_object_config)
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

def analyze_trajectory(robot, joints, target_poses, config):
    # joints: (T, D), target_poses: (T, 7)
    T = joints.shape[0]
    # Vectorized forward kinematics
    def get_ee_pose(cfg):
        fk_poses_arr = robot.forward_kinematics(cfg=cfg)
        ee_pose_in_root_arr = fk_poses_arr[robot.links.names.index("end_effector")]
        return ee_pose_in_root_arr
    get_ee_pose_vmap = jax.vmap(get_ee_pose, in_axes=0)
    ee_poses = get_ee_pose_vmap(joints)  # (T, 7)
    # Convert to SE3 objects
    ee_se3s = se3_from_pose_vmap(ee_poses)
    target_se3s = se3_from_pose_vmap(target_poses)
    # Compute error
    def calc_error(target_se3, ee_se3):
        return (target_se3.inverse() @ ee_se3).log()
    calc_error_vmap = jax.vmap(calc_error)
    errors = calc_error_vmap(target_se3s, ee_se3s)  # (T, 6)
    position_errors = jnp.linalg.norm(errors[:, :3], axis=1)
    orientation_errors = jnp.linalg.norm(errors[:, 3:], axis=1)
    max_position_error = float(jnp.max(position_errors))
    max_orientation_error = float(jnp.max(orientation_errors))
    # Smoothness cost
    smoothness_costs = jnp.linalg.norm(joints[1:] - joints[:-1], axis=1) * config['weights']['smoothness']
    max_smoothness_cost = float(jnp.max(smoothness_costs))
    # Timesteps exceeding tolerance
    error_timesteps = jnp.where((position_errors > config['tolerance']['position_error']) |
                                (orientation_errors > config['tolerance']['orientation_error']))[0].tolist()
    return max_position_error, max_orientation_error, max_smoothness_cost, error_timesteps

def visualize_trajectory(server, urdf_vis, base_frame, Ts_world_root, joints, target_poses_se3, config, robot):
    num_timesteps = len(target_poses_se3)
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    current_error_text = server.gui.add_text("Current Error: ", "Position: 0.0000 m, Orientation: 0.0000 rad")
    status_text = server.gui.add_text("Status: ", "✅ PASSED")
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
        time.sleep(config['visualization']['sleep_time'])

@jax.jit
def solve_eetrack_optimization(
    robot: pk.Robot,
    target_poses: jnp.ndarray,  # (T, 7)
    weights: TrackingWeights,
) -> Tuple[tuple[jaxlie.SE3, ...], jnp.ndarray]:
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

    # Safely generate costs using for loop
    costs = []
    for t in range(timesteps):
        costs.append(path_tracking_cost_t(var_joints[t], target_poses[t]))
        costs.append(pk.costs.limit_cost(robot, var_joints[t], weights["joint_limits"]))
    for t in range(timesteps - 1):
        costs.append(smoothness_cost_t(var_joints[t+1], var_joints[t]))
    solution = (
        jaxls.LeastSquaresProblem(costs, [var_joints])
        .analyze()
        .solve()
    )
    solved_Ts_world_root = tuple([jaxlie.SE3.identity() for _ in range(timesteps)])
    solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])
    return solved_Ts_world_root, solved_joints

def main():
    config, asset_dir = load_config()
    robot, modified_urdf = load_robot(config)
    welding_path = get_welding_path(config, asset_dir, modified_urdf)
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
    )
    # Compute welding_object, parent_pose, welding_object_pose (restored as in old version)
    welding_object, welding_object_pose, parent_pose = None, None, None
    if config["welding_path_from_object"]:
        welding_object_config = config["welding_object"].copy()
        welding_object_pose = jaxlie.SE3(jnp.array(welding_object_config.pop("pose"))[None])
        welding_object_parent = welding_object_config.pop("parent", None)
        if welding_object_parent == "mid_sole_link":
            left_sole_link_pose = jaxlie.SE3.from_matrix(modified_urdf.get_transform("left_sole_link")[None])
            right_sole_link_pose = jaxlie.SE3.from_matrix(modified_urdf.get_transform("right_sole_link")[None])
            parent_pose = get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose)
        else:
            parent_pose = jaxlie.SE3.identity((1,))
        welding_object_pose = parent_pose @ welding_object_pose
        welding_object = WeldObject(**welding_object_config)
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
    Ts_world_root, joints = solve_eetrack_optimization(robot, target_poses, weights)
    # Error analysis based on SE3 object list (as in old version)
    max_position_error = 0.0
    max_orientation_error = 0.0
    error_timesteps = []
    for t in range(num_timesteps):
        robot_cfg = joints[t]
        end_effector_link_idx = robot.links.names.index("end_effector")
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
        T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
        T_world_root = jaxlie.SE3.identity()
        ee_pose = T_world_root @ T_root_ee
        error = (target_poses_se3[t].inverse() @ ee_pose).log()
        position_error = np.linalg.norm(error[:3])
        orientation_error = np.linalg.norm(error[3:])
        max_position_error = max(max_position_error, position_error)
        max_orientation_error = max(max_orientation_error, orientation_error)
        if (position_error > config['tolerance']['position_error'] or 
            orientation_error > config['tolerance']['orientation_error']):
            error_timesteps.append(t)
    position_failed = max_position_error > config['tolerance']['position_error']
    orientation_failed = max_orientation_error > config['tolerance']['orientation_error']
    print(f"=== Error Analysis ===")
    print(f"Max Position Error: {max_position_error:.4f} m (tolerance: {config['tolerance']['position_error']:.4f} m)")
    print(f"Max Orientation Error: {max_orientation_error:.4f} rad (tolerance: {config['tolerance']['orientation_error']:.4f} rad)")
    print(f"Timesteps with errors: {len(error_timesteps)}/{num_timesteps}")
    if position_failed or orientation_failed:
        print(f"❌ FAILED: {'Position' if position_failed else ''}{' and ' if position_failed and orientation_failed else ''}{'Orientation' if orientation_failed else ''} max error exceeded tolerance")
    else:
        print(f"✅ PASSED: All errors within tolerance")
    # Use target_poses_se3 in visualization loop
    visualize_trajectory(server, urdf_vis, base_frame, Ts_world_root, joints, target_poses_se3, config, robot)

if __name__ == "__main__":
    main()
