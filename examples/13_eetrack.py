import os
import time
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import viser
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import xml.etree.ElementTree as ET
from io import StringIO
import yaml
from eetrack.utils.weld_objects import WeldObject


class TrackingWeights(TypedDict):
    position_tracking: float
    orientation_tracking: float
    smoothness: float
    joint_limits: float
    rest_pose: float

def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + 
             right_sole_link_pose.rotation().log())/2
        ),
        translation=(
            left_sole_link_pose.translation() +
            right_sole_link_pose.translation()
        )/2,
    )

def main():
    # Load configuration from YAML file
    asset_dir = Path(__file__).parent / "eetrack"
    config_file = asset_dir / "config.yaml"
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    urdf_path = config['robot']['urdf_path']
    urdf_obj = yourdfpy.URDF.load(urdf_path)

    # Set joint position from the terminal state of sitting policy and the target height
    sit_terminal_states = onp.load(config['robot']['sit_terminal_states_path'])
    idx = onp.abs(sit_terminal_states["target_height"] - config['robot']['sit_target_height']).argmin()
    root_pose = sit_terminal_states["root_state"][idx, :7]
    root_vel = sit_terminal_states["root_state"][idx, 7:]
    joint_pos = sit_terminal_states["joint_state"][idx, 0]
    joint_vel = sit_terminal_states["joint_state"][idx, 1]
    lab2yourdf = [onp.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in urdf_obj.actuated_joint_names]
    urdf_obj.update_cfg(joint_pos[lab2yourdf])
    for joint in urdf_obj.robot.joints:
        if joint.name in urdf_obj.actuated_joint_names and joint.name not in config['robot']['movable_joints']:
            joint.type = "fixed"
            joint.origin = urdf_obj.get_transform(joint.child, joint.parent)
    modified_urdf = yourdfpy.URDF(urdf_obj.robot, mesh_dir=os.path.dirname(urdf_path))

    robot = pk.Robot.from_urdf(modified_urdf)

    welding_path_from_object = config["welding_path_from_object"]
    if config["welding_path_from_object"]:
        welding_object_config = config["welding_object"]
        welding_object_pose = jaxlie.SE3(
            jnp.array(welding_object_config.pop("pose"))[None] # Ensure shape is (N, 7)
        )
        welding_object_parent = welding_object_config.pop("parent", None)
        if welding_object_parent == "mid_sole_link":
            left_sole_link_pose = jaxlie.SE3.from_matrix(modified_urdf.get_transform("left_sole_link")[None])
            right_sole_link_pose = jaxlie.SE3.from_matrix(modified_urdf.get_transform("right_sole_link")[None])
            parent_pose = get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose)
        else:
            parent_pose = jaxlie.SE3.identity((1,))
        welding_object_pose = parent_pose @ welding_object_pose
        welding_object = WeldObject(**config["welding_object"])
        welding_path_se3 = welding_object.get_welding_path(welding_object_pose)
        # wxyz_xyz -> xyz_xyzw
        welding_path_pos = welding_path_se3.translation()
        welding_path_xyzw = jnp.roll(welding_path_se3.rotation().wxyz, shift=-1, axis=-1)
        welding_path = jnp.concat([welding_path_pos, welding_path_xyzw], axis=-1)[0]
    else:
        # Load or generate welding path data
        # This should be a sequence of SE(3) poses for the end-effector
        welding_path_file = asset_dir / "welding_path.npy"
        
        # if welding_path_file.exists():
        #     # Load pre-defined welding path
        #     welding_path = onp.load(welding_path_file)
        # else:
        #     # Generate a simple welding path for demonstration
        #     # This should be replaced with actual welding path data
        #     num_timesteps = config['welding_path']['num_timesteps']
        #     welding_path = generate_demo_welding_path(config['welding_path'])
        #     onp.save(welding_path_file, welding_path)

        num_timesteps = config['welding_path']['num_timesteps']
        welding_path = generate_demo_welding_path(config['welding_path'])
        onp.save(welding_path_file, welding_path)

    target_poses = [
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(welding_path[i, 3:]),
            welding_path[i, :3]
        )
        for i in range(welding_path.shape[0])
    ]
    timesteps = len(target_poses)
    num_timesteps = timesteps

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, modified_urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    if welding_path_from_object:
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
    
    # Add error display
    current_error_text = server.gui.add_text("Current Error: ", "Position: 0.0000 m, Orientation: 0.0000 rad")
    status_text = server.gui.add_text("Status: ", "✅ PASSED")

    weights = pk.viewer.WeightTuner(
        server,
        TrackingWeights(  # type: ignore
            position_tracking=config['weights']['position_tracking'],
            orientation_tracking=config['weights']['orientation_tracking'],
            smoothness=config['weights']['smoothness'],
            joint_limits=config['weights']['joint_limits'],
            rest_pose=config['weights']['rest_pose'],
        ),
    )

    Ts_world_root, joints = None, None

    def generate_trajectory():
        nonlocal Ts_world_root, joints
        gen_button.disabled = True
        Ts_world_root, joints = solve_eetrack_optimization(
            robot=robot,
            target_poses=target_poses,
            weights=weights.get_weights(),
        )
        gen_button.disabled = False

    gen_button = server.gui.add_button("Optimize!")
    gen_button.on_click(lambda _: generate_trajectory())

    generate_trajectory()
    assert Ts_world_root is not None and joints is not None

    # Initialize error tracking
    error_timesteps = []

    # Calculate errors for all timesteps
    max_position_error = 0.0
    max_orientation_error = 0.0
    
    for t in range(num_timesteps):
        # Get current end effector pose
        robot_cfg = joints[t]
        end_effector_link_idx = robot.links.names.index("end_effector")
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
        T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
        T_world_root = jaxlie.SE3.identity()
        ee_pose = T_world_root @ T_root_ee
        
        # Calculate error
        error = (target_poses[t].inverse() @ ee_pose).log()
        position_error = onp.linalg.norm(error[:3])
        orientation_error = onp.linalg.norm(error[3:])
        
        # Update max errors
        max_position_error = max(max_position_error, position_error)
        max_orientation_error = max(max_orientation_error, orientation_error)
        
        # Check if error exceeds tolerance
        if (position_error > config['tolerance']['position_error'] or 
            orientation_error > config['tolerance']['orientation_error']):
            error_timesteps.append(t)

    # Check max error tolerance
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

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            
            # Calculate current error
            robot_cfg = joints[tstep]
            end_effector_link_idx = robot.links.names.index("end_effector")
            fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
            ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
            T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
            T_world_root = jaxlie.SE3.identity()
            ee_pose = T_world_root @ T_root_ee
            
            error = (target_poses[tstep].inverse() @ ee_pose).log()
            current_position_error = onp.linalg.norm(error[:3])
            current_orientation_error = onp.linalg.norm(error[3:])
            
            # Update error displays
            current_error_text.value = f"Position: {current_position_error:.4f} m, Orientation: {current_orientation_error:.4f} rad"
            
            # Update status based on current error
            if (current_position_error > config['tolerance']['position_error'] or 
                current_orientation_error > config['tolerance']['orientation_error']):
                status_text.value = "❌ ERROR: Current error exceeds tolerance"
            elif position_failed or orientation_failed:
                status_text.value = "❌ FAILED: Max error exceeded tolerance"
            else:
                status_text.value = "✅ PASSED: All errors within tolerance"
            
            base_frame.wxyz = onp.array(Ts_world_root[tstep].wxyz_xyz[:4])
            base_frame.position = onp.array(Ts_world_root[tstep].wxyz_xyz[4:])
            urdf_vis.update_cfg(onp.array(joints[tstep]))
            
            server.scene.add_frame(
                "/target_pose",
                axes_length=0.1,
                axes_radius=0.002,
                wxyz=target_poses[tstep].rotation().wxyz,
                position=target_poses[tstep].translation(),
            )

        time.sleep(config['visualization']['sleep_time'])


def generate_demo_welding_path(welding_config: dict) -> onp.ndarray:
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


@jdc.jit
def solve_eetrack_optimization(
    robot: pk.Robot,
    target_poses: list[jaxlie.SE3],
    weights: TrackingWeights,
) -> Tuple[tuple[jaxlie.SE3, ...], jnp.ndarray]:

    timesteps = len(target_poses)

    # Robot properties
    available_joints = []
    for name in ["left_hip_yaw_joint", "right_hip_yaw_joint", "torso_joint"]:
        if name in robot.joints.actuated_names:
            available_joints.append(robot.joints.actuated_names.index(name))
    joints_to_move_less = jnp.array(available_joints)

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
    

    # modify rest_weights so that other parts except right arm joints are not moved

    costs: list[jaxls.Cost] = []
    rest_weights = jnp.full(var_joints.default_factory().shape, weights["rest_pose"])
    
    # Set higher rest pose weights for non-right-arm joints to keep them in place
    right_arm_joint_names = [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint"
    ]
    
    for i, joint_name in enumerate(robot.joints.actuated_names):
        if joint_name not in right_arm_joint_names:
            rest_weights = rest_weights.at[:, i].set(weights["rest_pose"] * 100.0)  # Higher weight for non-arm joints

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
            pk.costs.rest_cost(
                var_joints[t],
                var_joints.default_factory(),
                rest_weights
            )
        )
    
    for t in range(timesteps - 1):
        costs.append(
            smoothness_cost_t(
                var_joints[t + 1],
                var_joints[t],
            )
        )

    solution = (
        jaxls.LeastSquaresProblem(costs, [var_joints])
        .analyze()
        .solve()
    )

    solved_Ts_world_root = tuple([jaxlie.SE3.identity() for _ in range(timesteps)])
    
    solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])

    return solved_Ts_world_root, solved_joints


if __name__ == "__main__":
    main()
