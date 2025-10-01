import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy
import torch
import trimesh
from glob import glob
from tqdm import tqdm
import argparse

from eetrack.utils.weld_objects import WeldObject
from pyroki.collision import CollisionDetector


def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    """Calculates the average pose between the left and right sole links of the
    robot. This is used as a stable reference frame.

    Args:
        left_sole_link_pose (jaxlie.SE3): The pose of the left sole link.
        right_sole_link_pose (jaxlie.SE3): The pose of the right sole link.

    Returns:
        (jaxlie.SE3): The average pose, computed by averaging the translation
            and rotation components.
    """
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + right_sole_link_pose.rotation().log()) / 2
        ),
        translation=(left_sole_link_pose.translation() + right_sole_link_pose.translation()) / 2,
    )


def load_config():
    """Loads the main configuration file (`config_solo.yaml`) for the script.

    Returns:
        Tuple[dict, Path]: A tuple containing:
            config (dict): The loaded configuration parameters.
            asset_dir (Path): The path to the `eetrack` asset directory.
    """
    asset_dir = Path(__file__).parent / "eetrack"
    config_file = asset_dir / "config_solo.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config, asset_dir


def load_robot(config):
    """Loads the robot models required for kinematics and collision checking.

    Args:
        config (dict): The main configuration dictionary.

    Returns:
        Tuple[pk.Robot, yourdfpy.URDF, yourdfpy.URDF]: A tuple containing:
            robot (pk.Robot): The `pyroki` robot object for kinematics.
            urdf (yourdfpy.URDF): The standard `yourdfpy` object with collision
                meshes.
            modified_urdf (yourdfpy.URDF): A `yourdfpy` object with non-movable
                joints fixed to a default sitting posture.
    """
    urdf_path = config['robot']['urdf_path']
    urdf = yourdfpy.URDF.load(
        urdf_path, force_mesh=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    robot = pk.Robot.from_urdf(urdf)

    modified_urdf = yourdfpy.URDF.load(urdf_path)
    sit_terminal_states = np.load(config['robot']['sit_terminal_states_path'])
    idx = np.abs(sit_terminal_states["target_height"] - config['robot']['sit_target_height']).argmin()
    joint_pos = sit_terminal_states["joint_state"][idx, 0]
    lab2yourdf = [np.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in modified_urdf.actuated_joint_names]
    modified_urdf.update_cfg(joint_pos[lab2yourdf])
    for joint in modified_urdf.robot.joints:
        if joint.name in modified_urdf.actuated_joint_names and joint.name not in config['robot']['movable_joints']:
            joint.type = "fixed"
            joint.origin = modified_urdf.get_transform(joint.child, joint.parent)
    modified_urdf = yourdfpy.URDF(modified_urdf.robot, mesh_dir=Path(urdf_path).parent, build_collision_scene_graph=True, load_collision_meshes=True)

    return robot, urdf, modified_urdf


def load_scene_mesh(config, z_height):
    """Loads the obstacle and welding object to create a complete scene mesh.

    Args:
        config (dict): The main configuration dictionary.
        z_height (float): The z-height to apply to the scene.

    Returns:
        Tuple[trimesh.Trimesh, WeldObject, jaxlie.SE3]: A tuple containing:
            scene_mesh (trimesh.Trimesh): The combined mesh of the environment.
            welding_object (WeldObject): The welding object instance.
            welding_object_pose (jaxlie.SE3): The pose of the welding object.
    """
    welding_object_config = config["welding_object"].copy()
    welding_object_config.pop('pose', None)
    welding_object_config.pop('yaw', None)
    welding_object_config.pop("parent", None)
    welding_object = WeldObject(**welding_object_config)

    welding_object_pose = jaxlie.SE3(jnp.array([[1.,0.,0.,0.,0.,0.,0.]]))

    obstacle_cfg = config["obstacle"]
    if obstacle_cfg["mesh_path"] is not None:
        scene = trimesh.load(obstacle_cfg["mesh_path"], force="scene")
    elif obstacle_cfg["urdf_path"] is not None:
        obstacle_urdf = yourdfpy.URDF.load(obstacle_cfg["urdf_path"])
        scene = obstacle_urdf.scene

    scene.apply_translation([0,0,-z_height])
    scene.add_geometry(welding_object.trimesh, "welding_object")
    scene_mesh = scene.to_mesh()

    return scene_mesh, welding_object, welding_object_pose


def check_sitting_collision(config, robot, urdf, scene_mesh, data, z_height):
    """Performs the core batch collision checking for the sitting motion.

    Args:
        config (dict): The main configuration dictionary.
        robot (pk.Robot): The pyroki robot model.
        urdf (yourdfpy.URDF): The robot's URDF model.
        scene_mesh (trimesh.Trimesh): The mesh of the environment.
        data (dict): The loaded data from the `.pt` file, containing successful
            samples.
        z_height (float): The z-height of the welding object.

    Returns:
        Tuple: A tuple containing:
            is_collide (jnp.ndarray): Boolean array indicating collision for
                every timestep of every trajectory.
            is_collision_sample (jnp.ndarray): Boolean array indicating if a
                sample has a collision at any point.
            mid_sole_poses (jaxlie.SE3): The poses of the mid_sole_link for
                each trajectory.
            joint_traj (jnp.ndarray): The joint trajectories for all samples.
            N (int): The number of successful samples checked.
            T (int): The number of timesteps in the sitting trajectory.
    """
    success = data["success"]
    N = success.sum().item()
    mid_sole_xyyaws = data["xyyaw_samples"][success].numpy()
    mid_sole_poses = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_rpy_radians(jnp.zeros(N), jnp.zeros(N), mid_sole_xyyaws[:,2]),
        translation=jnp.concat([mid_sole_xyyaws[:,:2], -z_height*jnp.ones((N,1))], axis=1),
    )

    coll_det = CollisionDetector(robot, urdf, scene_mesh)
    check_collision_batch_fn = jax.vmap(coll_det.check_collision, in_axes=0)

    sit_traj = np.load(config['robot']['sit_terminal_states_path'])
    idx = np.abs(sit_traj["target_height"] - config['robot']['sit_target_height']).argmin()
    lab2yourdf = [np.where(sit_traj["lab_joint"] == jn)[0].item() for jn in urdf.actuated_joint_names]

    T = sit_traj["steps"][idx] + 1
    joint_traj = sit_traj["joint_traj"][idx,:T,0][None,:,lab2yourdf].repeat(N,0).reshape(N*T,-1)
    mid_sole_poses = jaxlie.SE3(mid_sole_poses.parameters()[:,None].repeat(T,1).reshape(N*T,-1))

    B = 5000
    is_collide = []
    for i in tqdm(range(N*T//B + 1)):
        ids = jnp.arange(B*i, min(B*(i+1), N*T))
        is_collide.append(check_collision_batch_fn(joint_traj[ids], jaxlie.SE3(mid_sole_poses.parameters()[ids])))
    is_collide = jnp.concat(is_collide)
    is_collision_sample = is_collide.reshape(N,T).any(axis=1)

    return is_collide, is_collision_sample, mid_sole_poses, joint_traj, N, T


def visualize_trajectory(server, base_frame, urdf_vis, mid_sole_frame, root_traj, joint_traj, mid_sole_traj, is_collide):
    """Sets up the Viser GUI and runs the interactive trajectory visualization.

    Args:
        server (viser.ViserServer): The Viser server instance.
        base_frame (viser.scene.FrameHandle): The handle for the robot's base
            frame.
        urdf_vis (ViserUrdf): The Viser URDF visualizer instance.
        mid_sole_frame (viser.scene.FrameHandle): The handle for the mid sole
            frame.
        root_traj (jaxlie.SE3): The SE3 trajectory of the robot's root.
        joint_traj (jnp.ndarray): The joint trajectory.
        mid_sole_traj (jaxlie.SE3): The SE3 trajectory of the mid sole frame.
        is_collide (jnp.ndarray): Boolean array indicating collision for each
            timestep.
    """
    N, T = joint_traj.shape[:2]
    playing = server.gui.add_checkbox("playing", True)
    sample_slider = server.gui.add_slider("samples", 0, N - 1, 1, 0)
    timestep_slider = server.gui.add_slider("timestep", 0, T - 1, 1, 0)
    status_text = server.gui.add_text("Status: ", "")
    
    while True:
        with server.atomic():
            if playing.value:
                if timestep_slider.value + 1 == T:
                    sample_slider.value = (sample_slider.value + 1) % N
                timestep_slider.value = (timestep_slider.value + 1) % T
            nstep = sample_slider.value
            tstep = timestep_slider.value
            urdf_vis.update_cfg(np.array(joint_traj[nstep,tstep]))
            base_frame.wxyz = np.array(root_traj.rotation().wxyz[nstep,tstep])
            base_frame.position = np.array(root_traj.translation()[nstep,tstep])
            mid_sole_frame.wxyz = np.array(mid_sole_traj.rotation().wxyz[nstep,tstep])
            mid_sole_frame.position = np.array(mid_sole_traj.translation()[nstep,tstep])
            status_text.value = "❌ Collision" if is_collide[nstep,tstep] else "✅ No collision"
        time.sleep(0.05)


def main():
    """Main function to orchestrate the sitting collision check pipeline."""
    parser = argparse.ArgumentParser(description="Check sit base collision during sitting trajectories.")
    parser.add_argument('--sit_target_height', type=float, default=0.4, help='Sit target height (Ex: 0.4)')
    parser.add_argument('--z_height', type=float, default=0.3, help='z height of welding object (Ex: 0.1)')
    args = parser.parse_args()

    config, asset_dir = load_config()
    robot, urdf, modified_urdf = load_robot(config)

    data_path = Path(f"files/batch_pipeline_h{int(100*args.sit_target_height)}_z{int(1000*args.z_height)}/dummy_exp/batch_eetrack_results_inverse.pt")
    data = torch.load(data_path, weights_only=False)

    scene_mesh, welding_object, welding_object_pose = load_scene_mesh(config, args.z_height)

    is_collide, is_collision_sample, mid_sole_poses, joint_traj, N, T = check_sitting_collision(config, robot, urdf, scene_mesh, data, args.z_height)

    print(f"Collision rate: {100*is_collision_sample.sum()/N:.1f}% ({is_collision_sample.sum()}/{N})")

    success = data["success"]
    success_ids = success.nonzero().flatten()
    success_collision_ids = success_ids[np.array(is_collision_sample)]
    data["success"][success_collision_ids] = False

    save_dir = data_path.parents[0].with_name("sit_coll_filtered_exp")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / data_path.name
    torch.save(data, save_path)

    # Visualization
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    # Add welding_object, welding_object_pose, object_parent to viser.scene (as in old version)
    if welding_object is not None and welding_object_pose is not None:
        server.scene.add_mesh_trimesh("scene", scene_mesh)
        welding_object_frame = server.scene.add_frame(
            "welding_object_pose",
            axes_length=0.1,
            axes_radius=0.002,
            wxyz=welding_object_pose.rotation().wxyz[0],
            position=welding_object_pose.translation()[0],
        )
        mid_sole_frame = server.scene.add_frame(
            "/mid_sole_pose",
            axes_length=0.1,
            axes_radius=0.002,
        )

    fk = robot.forward_kinematics(joint_traj)
    left_sole_idx = robot.links.names.index("left_sole_link")
    right_sole_idx = robot.links.names.index("right_sole_link")
    left_sole_pose = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(fk[:,left_sole_idx,:4]),
        translation=fk[:,left_sole_idx,4:],
    )
    right_sole_pose = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(fk[:,right_sole_idx,:4]),
        translation=fk[:,right_sole_idx,4:],
    )
    mid_sole_pose = get_mid_sole_link_pose(left_sole_pose, right_sole_pose)
    root_traj = mid_sole_poses @ mid_sole_pose.inverse()

    visualize_trajectory(
        server,
        base_frame,
        urdf_vis,
        mid_sole_frame,
        jaxlie.SE3(root_traj.parameters().reshape(N,T,-1)),
        joint_traj.reshape(N,T,-1),
        jaxlie.SE3(mid_sole_poses.parameters().reshape(N,T,-1)),
        is_collide.reshape(N,T),
    )


if __name__ == "__main__":
    main()