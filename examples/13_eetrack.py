import os
import time
from pathlib import Path
from typing import Tuple, TypedDict

# Force CPU usage to avoid GPU memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

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
import numpy as np
from functools import partial

# Import refactored modules
from plotting import plot_optimization_iteration_costs
from optimization import (
    generate_demo_welding_path, 
    solve_eetrack_optimization_with_tracking,
    validate_trajectory,
    TrackingWeights
)
from viser_helpers import (
    setup_viser_gui, 
    setup_collision_visualization, 
    update_collision_bodies,
    update_visualization
)

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
    
    # Add robot collision detection with configurable collision ignore settings
    collision_config = config.get('collision', {})
    ignore_adjacent = collision_config.get('ignore_adjacent_links', True)
    ignore_pairs = collision_config.get('ignore_pairs', [])
    only_movable_links = collision_config.get('only_movable_links', True)
    exclude_links = collision_config.get('exclude_links', [])
    
    # Convert ignore_pairs to tuple of tuples for RobotCollision
    user_ignore_pairs = tuple(tuple(pair) for pair in ignore_pairs)
    
    # If only_movable_links is True, ignore collision pairs that don't involve movable links
    if only_movable_links:
        # Get all link names from URDF
        all_link_names = list(urdf_obj.link_map.keys())
        
        # Get movable link names (links connected to movable joints)
        movable_link_names = set()
        for joint_name in config['robot']['movable_joints']:
            if joint_name in urdf_obj.joint_map:
                joint = urdf_obj.joint_map[joint_name]
                # Add both parent and child links of movable joints
                movable_link_names.add(joint.parent)
                movable_link_names.add(joint.child)
        
        # Get fixed link names (all links except movable ones)
        fixed_link_names = set(all_link_names) - movable_link_names
        
        # Add excluded links to ignore_pairs (any collision pair involving excluded links)
        for excluded_link in exclude_links:
            if excluded_link in all_link_names:
                for other_link in all_link_names:
                    if excluded_link != other_link:
                        # Sort to avoid duplicate pairs
                        pair = sorted([excluded_link, other_link])
                        if pair not in ignore_pairs:
                            ignore_pairs.append(pair)
        
        # Only ignore collision pairs between remaining fixed links (neither link is movable)
        # This means we keep: (movable, movable) and (movable, remaining_fixed) pairs
        # We ignore: (remaining_fixed, remaining_fixed) pairs only
        remaining_fixed_links = fixed_link_names - set(exclude_links)
        for fixed_link1 in remaining_fixed_links:
            for fixed_link2 in remaining_fixed_links:
                if fixed_link1 != fixed_link2:
                    # Sort to avoid duplicate pairs
                    pair = sorted([fixed_link1, fixed_link2])
                    if pair not in ignore_pairs:
                        ignore_pairs.append(pair)
        
        # Update user_ignore_pairs
        user_ignore_pairs = tuple(tuple(pair) for pair in ignore_pairs)
    
    robot_coll = pk.collision.RobotCollision.from_urdf(
        urdf_obj,
        ignore_immediate_adjacents=ignore_adjacent,  # Ignore collisions between adjacent links
        user_ignore_pairs=user_ignore_pairs  # Additional pairs to ignore
    )
    
    print(f"RobotCollision created with {robot_coll.num_links} links and {len(robot_coll.active_idx_i)} active collision pairs")
    
    world_coll = []  # Empty list - no world collision constraints
    
    # Load or generate welding path data
    # This should be a sequence of SE(3) poses for the end-effector
    welding_path_file = asset_dir / "welding_path.npy"
    

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
    
    # Setup GUI controls
    gui_controls = setup_viser_gui(server, num_timesteps, config)
    
    # Setup collision visualization
    collision_mesh_handles, collision_cache = setup_collision_visualization(server, robot_coll)

    weights = pk.viewer.WeightTuner(
        server,
        TrackingWeights(  # type: ignore
            position_tracking=config['weights']['position_tracking'],
            orientation_tracking=config['weights']['orientation_tracking'],
            smoothness=config['weights']['smoothness'],
            joint_limits=config['weights']['joint_limits'],
            collision=config['weights']['collision'],
        ),
        min={
            "position_tracking": 0.0,
            "orientation_tracking": 0.0,
            "smoothness": 0.0,
            "joint_limits": 0.0,
            "collision": 0.0,
        },
        max={
            "position_tracking": 5000.0,
            "orientation_tracking": 5000.0,
            "smoothness": 2000.0,
            "joint_limits": 5000.0,
            "collision": 100.0,
        },
        step={
            "position_tracking": 1.0,
            "orientation_tracking": 1.0,
            "smoothness": 1.0,
            "joint_limits": 1.0,
            "collision": 1.0,
        },
    )

    Ts_world_root, joints = None, None
    optimization_history = None

    def generate_trajectory():
        nonlocal Ts_world_root, joints, optimization_history, analysis_results
        gen_button.disabled = True
        print("Starting optimization...")
        
        # Run optimization
        Ts_world_root, joints, optimization_history = solve_eetrack_optimization_with_tracking(
            robot=robot,
            target_poses=target_poses,
            weights=weights.get_weights(),
            robot_coll=robot_coll,
            world_coll=world_coll,
            safety_margin=config.get('collision', {}).get('safety_margin', 0.05),
        )
        
        print(f"Optimization completed!")
        print(f"Final cost: {optimization_history['final_cost']:.6f}")
        print(f"Converged: {optimization_history['converged']}")
        
        # Analyze trajectory
        analysis_results = validate_trajectory(joints, robot, target_poses, robot_coll, config)
        
        # Print validation results
        print("\n=== Trajectory Validation Results ===")
        for test, result in analysis_results['validation_summary'].items():
            print(f"{test}: {result}")
        
        # Plot optimization iteration progress with detailed analysis
        print("\nGenerating detailed optimization iteration progress plots...")
        plot_optimization_iteration_costs(optimization_history, config)
        
        gen_button.disabled = False

    gen_button = server.gui.add_button("Optimize!")
    gen_button.on_click(lambda _: generate_trajectory())
    

    
    def plot_iteration_costs():
        if optimization_history is not None:
            print("\nGenerating detailed optimization iteration progress plots...")
            plot_optimization_iteration_costs(optimization_history, config)
        else:
            print("No optimization history available. Please run optimization first.")
    
    iteration_plot_button = server.gui.add_button("Plot Iteration Costs")
    iteration_plot_button.on_click(lambda _: plot_iteration_costs())

    # Initialize variables
    analysis_results = None
    target_points = onp.array([target_poses[t].translation() for t in range(num_timesteps)])
    target_colors = onp.tile(onp.array(config['visualization']['target_color']), (num_timesteps, 1))

    while True:
        with server.atomic():
            # Check if optimization has been completed
            if joints is None:
                # Show waiting message
                gui_controls['error_text'].value = "Waiting for optimization..."
                gui_controls['status_text'].value = "⏳ Please click 'Optimize!' to start"
                gui_controls['collision_text'].value = "⏳ Waiting for optimization"
                gui_controls['min_distance_text'].value = "⏳ Waiting for optimization"
                # Show target path only
                server.scene.add_point_cloud(
                    "/target_path",
                    target_points[gui_controls['timestep_slider'].value:gui_controls['timestep_slider'].value+1],
                    target_colors[gui_controls['timestep_slider'].value:gui_controls['timestep_slider'].value+1],
                    point_size=config['visualization']['point_size'],
                )
            else:
                tstep = gui_controls['timestep_slider'].value
                gui_controls['status_text'].value = "✅ PASSED: All errors within tolerance"
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
                # Update timestep
                if gui_controls['playing'].value:
                    gui_controls['timestep_slider'].value = (gui_controls['timestep_slider'].value + 1) % num_timesteps
                # Update visualization
                update_visualization(
                    tstep, joints, Ts_world_root, target_points, target_colors,
                    analysis_results['all_position_errors'] if analysis_results else [],
                    analysis_results['all_orientation_errors'] if analysis_results else [],
                    analysis_results['all_min_distances'] if analysis_results else [],
                    analysis_results['all_collision_violations'] if analysis_results else [],
                    analysis_results['all_collision_distances'] if analysis_results else [],
                    analysis_results['position_failed'] if analysis_results else False,
                    analysis_results['orientation_failed'] if analysis_results else False,
                    base_frame, urdf_vis, server, gui_controls, config
                )
                # Update collision body visualization
                update_collision_bodies(joints[tstep], robot_coll, collision_mesh_handles, collision_cache, gui_controls, robot)
        time.sleep(config['visualization']['sleep_time'])


if __name__ == "__main__":
    main()
