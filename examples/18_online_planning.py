"""Online Planning

Run online planning in collision aware environments.
"""
import os
import time
import yaml
import trimesh
import yourdfpy
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, Sphere, Capsule
from pyroki.collision._robot_collision_custom import RobotCollisionV2
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
from itertools import product
from scipy.interpolate import splprep, splev

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import pyroki_snippets as pks
from eetrack.utils.weld_objects import WeldObject


def solve_ik_batch(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz_batch: np.ndarray,
    target_position_batch: np.ndarray,
) -> np.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    target_link_index = jnp.array(robot.links.names.index(target_link_name))
    target_wxyz_batch = jnp.array(target_wxyz_batch)
    target_position_batch = jnp.array(target_position_batch)
    @jax.jit
    def _solve_ik_batch(target_wxyz, target_position):
        return pks._solve_ik._solve_ik_jax(
            robot,
            target_link_index,
            target_wxyz,
            target_position,
        )
    _solve_ik_jax_vmap = jax.vmap(_solve_ik_batch, in_axes=0)
    cfg = _solve_ik_jax_vmap(target_wxyz_batch, target_position_batch)
    return np.array(cfg)

def load_config():
    asset_dir = Path(__file__).parent / "eetrack"
    config_file = asset_dir / "config_solo.yaml"
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
    yourdf2lab = [urdf_obj.actuated_joint_names.index(jn) for jn in sit_terminal_states["lab_joint"]]
    urdf_obj.update_cfg(joint_pos[lab2yourdf])

    collision_cfg = config.get('collision', {})
    ignore_pairs = list(tuple(pair) for pair in collision_cfg.get('ignore_pairs', []))
    exclude_links = collision_cfg.get('exclude_links', [])
    ignore_pairs += list(product(exclude_links, exclude_links))

    robot_collision = RobotCollision.from_urdf(
        urdf_obj,
        user_ignore_pairs=ignore_pairs,
        ignore_immediate_adjacents=collision_cfg.get('ignore_adjacent_links', True),
    )
    return pk.Robot.from_urdf(urdf_obj), urdf_obj, robot_collision, joint_pos[lab2yourdf], joint_pos, yourdf2lab

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

def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + right_sole_link_pose.rotation().log()) / 2
        ),
        translation=(left_sole_link_pose.translation() + right_sole_link_pose.translation()) / 2,
    )

def get_welding_object_and_pose(config, urdf, sampled_x=None, sampled_y=None, sampled_yaw=None, sampled_z=None):
    welding_object_config = config["welding_object"].copy()
    welding_object_config.pop('pose', None)
    welding_object_config.pop('yaw', None)
    welding_object_config.pop("parent", None)

    left_sole = jaxlie.SE3.from_matrix(urdf.get_transform("left_sole_link")[None])
    right_sole = jaxlie.SE3.from_matrix(urdf.get_transform("right_sole_link")[None])
    mid_sole_pose_pelvis = get_mid_sole_link_pose(left_sole, right_sole)
    mid_sole_pos_obj = jnp.array([
        sampled_x,
        sampled_y,
        -sampled_z,
    ])
    mid_sole_so3_obj = jaxlie.SO3.from_rpy_radians(0.0, 0.0, sampled_yaw)
    mid_sole_pose_obj = jaxlie.SE3.from_rotation_and_translation(mid_sole_so3_obj, mid_sole_pos_obj)

    welding_object_pose = mid_sole_pose_pelvis @ mid_sole_pose_obj.inverse()
    welding_object = WeldObject(**welding_object_config)
    return welding_object, welding_object_pose, mid_sole_pose_pelvis

def get_welding_path(config, asset_dir, modified_urdf, sampled_x=None, sampled_y=None, sampled_yaw=None, sampled_z=None):
    welding_object, welding_object_pose, parent_pose = get_welding_object_and_pose(
        config, modified_urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)
    welding_path_se3 = welding_object.get_welding_path(welding_object_pose)
    welding_path_pos = welding_path_se3.translation()
    welding_path_xyzw = jnp.roll(welding_path_se3.rotation().wxyz, shift=-1, axis=-1)
    welding_path = jnp.concatenate([welding_path_pos, welding_path_xyzw], axis=-1)[0]
    return welding_path

def create_bspline_traj_from_points(points, size):
    tck, _ = splprep(points.T)
    spl_points = np.array(splev(np.linspace(0, 1, size), tck)).T
    # trimesh.Scene([trimesh.PointCloud(spl_points), trimesh.PointCloud(points, [255,0,0])]).show()
    return spl_points

def main():
    """Main function for online planning with collision."""
    config, asset_dir = load_config()
    collision_pairs = config.get('collision_pairs', [])
    robot, urdf, robot_coll, default_joint_pos, default_joint_pos_lab, yourdf2lab = load_robot(config)
    target_link_name = "end_effector"

    sampled_x, sampled_y, sampled_yaw, sampled_z = sample_welding_object_pose(config)
    if config["welding_path_from_object"]:
        welding_object, welding_object_pose, parent_pose = get_welding_object_and_pose(
            config, urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)
    welding_path = get_welding_path(config, asset_dir, urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)
    welding_path_se3 = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_quaternion_xyzw(welding_path[:,3:]),
        translation=welding_path[:,:3],
    )
    obstacle_urdf = yourdfpy.URDF.load(config["obstacle"]["urdf_path"])
    obstacle_coll_urdf = RobotCollision.from_urdf(obstacle_urdf)
    obstacle_coll = obstacle_coll_urdf.coll
    z_down_se3 = jaxlie.SE3.from_translation(np.array([0,0,-sampled_z]))
    obstacle_coll = obstacle_coll.transform(z_down_se3)
    # obstacle_coll.to_trimesh().show()
    obstacle_link_T = []
    for link_name in obstacle_coll_urdf.link_names:
        link_T = np.array(obstacle_urdf.get_transform(link_name))
        obstacle_link_T.append(link_T)
    obstacle_link_T = jnp.stack(obstacle_link_T)
    obstacle_link_se3 = jaxlie.SE3.from_matrix(obstacle_link_T)
    obstacle_coll = obstacle_coll.transform(obstacle_link_se3)
    # obstacle_coll = obstacle_coll.transform(welding_object_pose)

    N = 40
    radius = np.array(N*[0.0025])
    height = np.array(N*[0.15])
    pos = np.zeros((N,3))
    pos[:N//2,0] = np.linspace(-0.0475,0.0475,N//2)
    pos[:N//2,2] = 0.0025
    pos[N//2:,2] = np.linspace(0.0075,0.1025,N//2)
    wxyz = jaxlie.SO3.from_x_radians(np.array(N*[np.pi/2])).wxyz
    welding_object_coll = Capsule.from_radius_height(radius=radius, height=height, position=pos, wxyz=wxyz)
    # welding_object_coll = welding_object_coll.transform(welding_object_pose)
    # trimesh.Scene([welding_object_coll.to_trimesh(), welding_object.trimesh]).show()
    scene_coll = jax.tree.map(lambda x, y: jnp.concat([x, y]), obstacle_coll, welding_object_coll)

    # Define the online planning parameters.
    len_traj, dt = 100, 0.02

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(default_joint_pos)

    # Create interactive controller for IK target.
    urdf.update_cfg(default_joint_pos)
    init_ee_T = urdf.get_transform("end_effector")
    # ik_target_handle = server.scene.add_transform_controls(
    #     "/ik_target", scale=0.2, position=init_ee_T[:3,3], wxyz=jaxlie.SO3.from_matrix(init_ee_T[:3,:3]).wxyz
    # )

    reset_button = server.gui.add_button("Reset")
    scene_vis = server.scene.add_mesh_trimesh("/scene", mesh=scene_coll.to_trimesh())
    # welding_object_vis = server.scene.add_mesh_trimesh("/welding_object", mesh=welding_object_coll.to_trimesh())
    init_frame_handle = server.scene.add_batched_axes(
        "init_frame",
        axes_length=0.05,
        axes_radius=0.005,
        batched_positions=np.zeros((len_traj, 3)),
        batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * len_traj),
    )
    target_frame_handle = server.scene.add_batched_axes(
        "target_frame",
        axes_length=0.05,
        axes_radius=0.005,
        batched_positions=np.zeros((len_traj, 3)),
        batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * len_traj),
    )

    coll_world = robot_coll.coll
    # Reduce capsule size for welder.
    cap_list = []
    for i in range(len(coll_world.size)):
        cap_list.append(jax.tree.map(lambda x: x[i], coll_world))
    welder_cap = cap_list[-2]
    new_welder_cap = Capsule.from_radius_height(
        0.8*welder_cap.radius,
        welder_cap.height,
        welder_cap.pose.translation() + np.array([0.0025, 0, 0.004]),
        (welder_cap.pose.rotation() @ jaxlie.SO3.from_y_radians(np.radians(5.0))).wxyz,
    )
    cap_list[-2] = new_welder_cap
    # trimesh.Scene([welder_cap.to_trimesh(), trimesh.creation.axis(transform=welder_cap.pose.as_matrix()), new_welder_cap.to_trimesh()]).show()
    coll_world = jax.tree.map(lambda *args: jnp.stack(args), *cap_list)
    # fk_poses_arr = robot.forward_kinematics(cfg=default_joint_pos[-7:])
    # coll_world.transform_from_wxyz_position(fk_poses_arr[:,:4], fk_poses_arr[:,4:])
    robot_coll = RobotCollision(
        robot_coll.num_links,
        robot_coll.link_names,
        coll_world,
        robot_coll.active_idx_i,
        robot_coll.active_idx_j,
    )
    link_indices_for_collision = [robot.links.names.index(name) for name in robot_coll.link_names]
    # Update each collision capsule
    link_coll_vis = {}
    fk_poses_arr = robot.forward_kinematics(cfg=default_joint_pos)
    fk_results_collision = fk_poses_arr[jnp.array(link_indices_for_collision)]
    for i, link_name in enumerate(robot_coll.link_names):
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
        link_coll_vis[link_name] = server.scene.add_mesh_trimesh(
            f"collision/{link_name}",
            wireframe_mesh,
        )
        link_coll_vis[link_name].position = fk_results_collision[i, 4:]
        link_coll_vis[link_name].wxyz = fk_results_collision[i, :4]

    timing_handle = server.gui.add_number("Solve time (ms)", 0.001, disabled=True)

    data_dict = {
        "init_joint_pos_lab": [],
        "root_pos_obj": [],
        "root_wxyz_obj": [],
        "target_joint_pos_traj_lab": [],
        "target_ee_pos_traj": [],
        "target_ee_wxyz_traj": [],
    }
    for _ in range(100):
        sampled_x, sampled_y, sampled_yaw, sampled_z = sample_welding_object_pose(config)
        if config["welding_path_from_object"]:
            welding_object, welding_object_pose, parent_pose = get_welding_object_and_pose(
                config, urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)
        welding_path = get_welding_path(config, asset_dir, urdf, sampled_x, sampled_y, sampled_yaw, sampled_z)
        welding_path_se3 = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.from_quaternion_xyzw(welding_path[:,3:]),
            translation=welding_path[:,:3],
        )

        obstacle_coll_curr = obstacle_coll.transform(welding_object_pose)
        welding_object_coll_curr = welding_object_coll.transform(welding_object_pose)
        world_coll_list = [obstacle_coll_curr, welding_object_coll_curr]
        scene_coll_curr = scene_coll.transform(welding_object_pose)

        sol_pos, sol_wxyz = None, None
        sol_traj = np.array(
            robot.joint_var_cls.default_factory()[None].repeat(len_traj, axis=0)
        )
        # sol_traj[:] = default_joint_pos[-7:]
        # sol_traj[0] = default_joint_pos[-7:]
        # Joint space interpolation
        # welding_init_joint_pos = pks.solve_ik_with_collision(
        #     robot,
        #     robot_coll,
        #     scene_coll_curr,
        #     target_link_name,
        #     welding_path_se3.translation()[0],
        #     welding_path_se3.rotation().wxyz[0],
        #     collision_pairs=collision_pairs,
        # )
        # sol_traj[-1] = welding_init_joint_pos
        # init_sol_traj = np.linspace(sol_traj[0], sol_traj[-1], len_traj)

        # SE3 space interpolation
        init_ee_se3 = jaxlie.SE3.from_matrix(init_ee_T)
        to_welding_init_se3 = jaxlie.SE3.exp(
            jnp.linspace(
                init_ee_se3.log(),
                welding_path_se3.log()[0],
                len_traj,
            )
        )
        start_ee_pos = np.array(init_ee_se3.translation())
        end_ee_pos = np.array(welding_path_se3.translation()[0])
        waypoints = np.array([
            start_ee_pos,
            (0.75*start_ee_pos + 0.25*end_ee_pos) + [-0.075, 0.0, 0.05],
            (0.25*start_ee_pos + 0.75*end_ee_pos) + [-0.075, 0.0, 0.05],
            end_ee_pos,
        ])
        waypoints = create_bspline_traj_from_points(waypoints, len_traj)
        to_welding_init_se3 = jaxlie.SE3.from_rotation_and_translation(
            to_welding_init_se3.rotation(),
            waypoints,
        )
        init_frame_handle.batched_positions = np.array(to_welding_init_se3.translation())  # type: ignore[attr-defined]
        init_frame_handle.batched_wxyzs = np.array(to_welding_init_se3.rotation().wxyz)  # type: ignore[attr-defined]
        init_sol_traj = solve_ik_batch(
            robot,
            target_link_name,
            to_welding_init_se3.rotation().wxyz,
            to_welding_init_se3.translation(),
        )
        init_sol_traj[:,:-7] = default_joint_pos[:-7]
        sol_traj = init_sol_traj.copy()

        def button_callback():
            nonlocal sol_traj, init_sol_traj
            sol_traj = init_sol_traj.copy()
        reset_button.on_click(lambda _: button_callback())

        fixed_joint_ids = [i for i, jn in enumerate(urdf.actuated_joint_names) if jn not in config["robot"]["movable_joints"]]
        sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(
            robot=robot,
            robot_coll=robot_coll,
            world_coll=scene_coll_curr,
            target_link_name=target_link_name,
            # target_position=np.array(ik_target_handle.position),
            # target_wxyz=np.array(ik_target_handle.wxyz),
            target_position=welding_path_se3.translation()[0],
            target_wxyz=welding_path_se3.rotation().wxyz[0],
            timesteps=len_traj,
            dt=dt,
            start_cfg=sol_traj[0],
            prev_sols=sol_traj,
            fixed_joint_ids=fixed_joint_ids,
            default_joint_pos=default_joint_pos,
            waypoints=waypoints,
        )

        root_pose_obj = (welding_object_pose @ jaxlie.SE3.from_translation(np.array([0,0,-sampled_z]))).inverse()
        target_joint_pos_traj = np.concatenate([default_joint_pos_lab[:-7][None].repeat(len_traj,0), sol_traj], axis=1)

        data_dict["init_joint_pos_lab"].append(default_joint_pos_lab)
        data_dict["root_pos_obj"].append(root_pose_obj.translation()[0])
        data_dict["root_wxyz_obj"].append(root_pose_obj.rotation().wxyz[0])
        data_dict["target_joint_pos_traj_lab"].append(target_joint_pos_traj[:,yourdf2lab])
        data_dict["target_ee_pos_traj"].append(sol_pos[:,0])
        data_dict["target_ee_wxyz_traj"].append(sol_wxyz[:,0])

        # data_dict = {k: np.stack(v) for k, v in data_dict.items()}
        # np.savez("trajopt_result.npz", **data_dict)


        # Only for vmap. Too slow
        # solve_fn = pks.get_online_planning_solve_fn(
        #     robot=robot,
        #     robot_coll=robot_coll,
        #     target_link_name=target_link_name,
        #     timesteps=len_traj,
        #     dt=dt,
        # )
        # sol_traj, sol_pos, sol_wxyz = solve_fn(welding_path_se3.translation()[0], welding_path_se3.rotation().wxyz[0], sol_traj[0], sol_traj)
        # sol_traj = np.array(sol_traj)
        # sol_pos = np.array(sol_pos)
        # sol_wxyz = np.array(sol_wxyz)

        # solve_fn_vmap = jax.vmap(solve_fn, in_axes=(0,0,0,0,None))
        # solve_fn_vmap(welding_path_se3.translation()[0][None].repeat(50,0),welding_path_se3.rotation().wxyz[0][None].repeat(50,0),sol_traj[0][None].repeat(50,0), sol_traj[None].repeat(50,0), scene_coll_curr)


        scene_vis.position = welding_object_pose.translation()[0]
        scene_vis.wxyz = welding_object_pose.rotation().wxyz[0]
        # welding_object_vis.position = welding_object_pose.translation()[0]
        # welding_object_vis.wxyz = welding_object_pose.rotation().wxyz[0]
        t = 0
        while True:
            t = t % len_traj
            start_time = time.time()

            # sol_traj, sol_pos, sol_wxyz, summary = pks.solve_online_planning(
            #     robot=robot,
            #     robot_coll=robot_coll,
            #     world_coll=scene_coll_curr,
            #     target_link_name=target_link_name,
            #     # target_position=np.array(ik_target_handle.position),
            #     # target_wxyz=np.array(ik_target_handle.wxyz),
            #     target_position=welding_path_se3.translation()[0],
            #     target_wxyz=welding_path_se3.rotation().wxyz[0],
            #     timesteps=len_traj,
            #     dt=dt,
            #     start_cfg=sol_traj[0],
            #     prev_sols=sol_traj,
            #     collision_pairs=collision_pairs,
            # )

            # if reset_button.value:
            #     sol_traj = init_sol_traj.copy()

            # Update timing handle.
            timing_handle.value = (time.time() - start_time) * 1000

            # Update visualizer.
            urdf_vis.update_cfg(
                sol_traj[t]
            )  # The first step of the online trajectory solution.

            # Update the planned trajectory visualization.
            if hasattr(target_frame_handle, "batched_positions"):
                target_frame_handle.batched_positions = np.array(sol_pos)  # type: ignore[attr-defined]
                target_frame_handle.batched_wxyzs = np.array(sol_wxyz)  # type: ignore[attr-defined]
                # target_frame_handle.batched_positions = np.array(to_welding_init_se3.translation())  # type: ignore[attr-defined]
                # target_frame_handle.batched_wxyzs = np.array(to_welding_init_se3.rotation().wxyz)  # type: ignore[attr-defined]
            else:
                # This is an older version of Viser.
                target_frame_handle.positions_batched = np.array(sol_pos)  # type: ignore[attr-defined]
                target_frame_handle.wxyzs_batched = np.array(sol_wxyz)  # type: ignore[attr-defined]

            # Update collision capsules
            fk_poses_arr = robot.forward_kinematics(cfg=sol_traj[t])
            fk_results_collision = fk_poses_arr[jnp.array(link_indices_for_collision)]
            for i, link_name in enumerate(robot_coll.link_names):
                link_coll_vis[link_name].position = fk_results_collision[i, 4:]
                link_coll_vis[link_name].wxyz = fk_results_collision[i, :4]

            t += 1

            time.sleep(0.01)

            if t == len_traj:
                break


if __name__ == "__main__":
    main()
