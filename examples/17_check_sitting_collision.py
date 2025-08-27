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

from eetrack.utils.weld_objects import WeldObject
from pyroki.collision import CollisionDetector


def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + right_sole_link_pose.rotation().log()) / 2
        ),
        translation=(left_sole_link_pose.translation() + right_sole_link_pose.translation()) / 2,
    )


def load_config():
    asset_dir = Path(__file__).parent / "eetrack"
    config_file = asset_dir / "config_solo.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config, asset_dir


def load_robot(config):
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


def visualize_trajectory(server, base_frame, urdf_vis, mid_sole_frame, root_traj, joint_traj, mid_sole_traj, is_collide):
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
    config, asset_dir = load_config()
    robot, urdf, modified_urdf = load_robot(config)

    sampled_x, sampled_y, sampled_yaw, sampled_z = sample_welding_object_pose(config)

    sit_height_cm = int(100*config['robot']['sit_target_height'])
    sampled_z_mm = int(1000*sampled_z)
    data_paths = sorted(glob(f"files_temp/batch_pipeline_h*_z{sampled_z_mm}/dummy_exp/batch_eetrack_results_inverse.pt"))
    data_list = [torch.load(data_path, weights_only=False) for data_path in data_paths]
    success_rates = [data["success_rate"] for data in data_list]
    high_sr_idx = np.argmax(success_rates)
    data_path = Path(data_paths[high_sr_idx])
    data = data_list[high_sr_idx]

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

    scene.apply_translation([0.045, 0, -sampled_z])
    scene.add_geometry(welding_object.trimesh, "welding_object")
    scene_mesh = scene.to_mesh()

    success = data["success"]
    N = success.sum().item()
    mid_sole_xyyaws = data["xyyaw_samples"][success].numpy()
    mid_sole_poses = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_rpy_radians(jnp.zeros(N), jnp.zeros(N), mid_sole_xyyaws[:,2]),
        translation=jnp.concat([mid_sole_xyyaws[:,:2], -sampled_z*jnp.ones((N,1))], axis=1),
    )

    coll_det = CollisionDetector(robot, urdf, scene_mesh)
    check_collision_batch_fn = jax.vmap(coll_det.check_collision, in_axes=0)

    sit_traj = np.load(config['robot']['sit_terminal_states_path'])
    idx = np.abs(sit_traj["target_height"] - config['robot']['sit_target_height']).argmin()
    lab2yourdf = [np.where(sit_traj["lab_joint"] == jn)[0].item() for jn in urdf.actuated_joint_names]

    T = sit_traj["steps"][idx] + 1
    joint_traj = sit_traj["joint_traj"][idx,:T,0][None,:,lab2yourdf].repeat(N,0).reshape(N*T,-1)
    mid_sole_poses = jaxlie.SE3(mid_sole_poses.parameters()[:,None].repeat(T,1).reshape(N*T,-1))

    # coll_det.check_collision(joint_traj[0], jaxlie.SE3(mid_sole_poses.parameters()[0]))
    B = 5000
    is_collide = []
    for i in tqdm(range(N*T//B + 1)):
        ids = jnp.arange(B*i, min(B*(i+1), N*T))
        is_collide.append(check_collision_batch_fn(joint_traj[ids], jaxlie.SE3(mid_sole_poses.parameters()[ids])))
    is_collide = jnp.concat(is_collide)
    is_collision_sample = is_collide.reshape(N,T).any(axis=1)

    print(f"Collision rate: {100*is_collision_sample.sum()/N:.1f}% ({is_collision_sample.sum()}/{N})")

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

    # Simple hand reaching collision test
    # ee_idx = robot.links.names.index("end_effector")
    # last_ee_pos = (jaxlie.SE3(mid_sole_poses.parameters().reshape(N,T,-1)[:,-1]) @ jaxlie.SE3(mid_sole_pose.parameters().reshape(N,T,-1)[:,-1]).inverse() @ jaxlie.SE3(fk.reshape(N,T,*fk.shape[-2:])[:,-1,ee_idx])).translation()
    # init_welding_pos = welding_object.get_welding_path(welding_object_pose).translation()[:,0].repeat(N,0)
    # interp_line = last_ee_pos[~is_collision_sample] * jnp.linspace(0,1,201).reshape(-1,1,1) + init_welding_pos[~is_collision_sample] * jnp.linspace(1,0,201).reshape(-1,1,1)
    # signed_dist, _ = coll_det.mesh_query_point(interp_line.reshape(-1,3))
    # is_line_collide = signed_dist < 0

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