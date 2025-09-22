import os
import sys
import time
import yaml
import yourdfpy
import numpy as np
from pathlib import Path
from typing import TypedDict
import threading
import trimesh
from itertools import product
from scipy.interpolate import splprep, splev

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

import pyroki as pk
import pyroki_snippets as pks
from pyroki.collision import RobotCollision, Capsule
from pyroki.collision._robot_collision_custom import RobotCollisionV2

import viser
from viser.extras import ViserUrdf

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from pyroki_ros.action import TrajOptSingleEE

from unitree_hg.msg import LowState

LAB_JOINT = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint', #11
    'right_shoulder_pitch_joint', #12
    'left_ankle_pitch_joint', #13
    'right_ankle_pitch_joint', #14
    'left_shoulder_roll_joint', #15
    'right_shoulder_roll_joint', #16
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint', #19
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint'
]


def body_pose(
    tf_buffer,
    frame: str,
    ref_frame: str = 'pelvis',
    stamp=None,
):
    """ --> tf does not exist """
    if stamp is None:
        stamp = rclpy.time.Time()
        # stamp = clock.get_time()
    try:
        # t = "ref{=pelvis}_from_frame" transform
        t = tf_buffer.lookup_transform(
            ref_frame,  # to
            frame,      # from
            stamp)
    except TransformException as ex:
        print(f'Could not transform {frame} to {ref_frame}: {ex}')
        raise

    txn = t.transform.translation
    rxn = t.transform.rotation

    xyz = np.array([txn.x, txn.y, txn.z])
    quat_wxyz = np.array([rxn.w, rxn.x, rxn.y, rxn.z])

    return xyz, quat_wxyz


# Vectorized SE3 conversion function
def se3_from_pose(pose):
    # pose: (7,) [x, y, z, x, y, z, w] (xyzw)
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_quaternion_xyzw(pose[3:]), pose[:3]
    )


def pose_stamped_msg_to_se3(pose_stamped_msg: PoseStamped):
    se3 = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(
            np.array([
                pose_stamped_msg.pose.orientation.w,
                pose_stamped_msg.pose.orientation.x,
                pose_stamped_msg.pose.orientation.y,
                pose_stamped_msg.pose.orientation.z,
            ])
        ),
        translation=np.array([
            pose_stamped_msg.pose.position.x,
            pose_stamped_msg.pose.position.y,
            pose_stamped_msg.pose.position.z,
        ]),
    )
    return se3


def create_bspline_traj_from_points(points, size):
    tck, _ = splprep(points.T)
    spl_points = np.array(splev(np.linspace(0, 1, size), tck)).T
    # trimesh.Scene([trimesh.PointCloud(spl_points), trimesh.PointCloud(points, [255,0,0])]).show()
    return spl_points


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


class TrajOptSingleEEActionServer(Node):
    def __init__(self):
        super().__init__('trajopt_single_ee_action_server')
        self.get_logger().info("Initializing trajopt_single_ee_action_server...")

        self._load_config()
        self._load_robot()
        self._load_collision_scene()

        self._init_visualizer()
        threading.Thread(target=self._vis_trajectory_callback, daemon=True).start()

        self._load_log()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._low_state = LowState()
        self._low_state_subscriber = self.create_subscription(
            LowState,
            "lowstate",
            self._low_state_cb,
            10,
        )

        self._warmup_jit_fn()

        self._action_server = ActionServer(
            self,
            TrajOptSingleEE,
            'trajopt_single_ee',
            self.execute_trajopt_single_ee,
        )

        self.get_logger().info("trajopt_single_ee_action_server starts!!!")


    def _low_state_cb(self, msg: LowState):
        self._low_state = msg


    def _load_config(self):
        self.asset_dir = Path(__file__).parent / "eetrack"
        config_file = self.asset_dir / "config_ros.yaml"
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)


    def _load_log(self):
        self.traj_data = np.load(self.config["log_path"], allow_pickle=True).item()["trajectories"]
        t_h = self.traj_data["timestamp_high_freq"]
        t_l = self.traj_data["timestamp_low_freq"]
        trajopt_start_low_idx = (self.traj_data["tasks"] == "trajopt").nonzero()[0][0]
        trajopt_start_high_idx = np.abs(t_h - t_l[trajopt_start_low_idx]).argmin()
        # Get joint pos
        lab2yourdf = [LAB_JOINT.index(jn) for jn in self.urdf_obj.actuated_joint_names]
        self.q = self.traj_data["q_traj"][trajopt_start_high_idx][lab2yourdf]
        # Compute welding object pose from log
        target_poses_w = self.traj_data["target_poses_w"]
        change_ids = np.flatnonzero((target_poses_w[1:] != target_poses_w[:-1]).any(axis=1))
        last_repeat_idx = change_ids[(change_ids > 0) & (target_poses_w[change_ids] == target_poses_w[change_ids-1]).all(axis=1)]
        start_ee_pose_w = self.traj_data["target_poses_w"][last_repeat_idx[-1]]
        end_ee_pose_w = self.traj_data["target_poses_w"][-1]
        start_ee_se3_w = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(start_ee_pose_w[3:]),
            start_ee_pose_w[:3]
        )
        end_ee_se3_w = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(end_ee_pose_w[3:]),
            end_ee_pose_w[:3]
        )
        mid_ee_pose_w = jaxlie.SE3.exp(
            (start_ee_se3_w.log() + end_ee_se3_w.log())/2
        )
        welding_object_pose_w = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_z_radians(mid_ee_pose_w.rotation().as_rpy_radians()[2]),
            mid_ee_pose_w.translation(),
        )
        root_pose_w = self.traj_data["root_states_w"][trajopt_start_low_idx]
        root_se3_w = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(root_pose_w[3:]),
            root_pose_w[:3],
        )
        self.welding_object_pose = root_se3_w.inverse() @ welding_object_pose_w


    def _load_robot(self):
        urdf_path = self.config['robot']['urdf_path']
        self.urdf_obj = yourdfpy.URDF.load(urdf_path)
        self.num_joints = self.urdf_obj.num_dofs

        self.mot2yourdf = [self.config["motor_joint"].index(jn) for jn in self.urdf_obj.actuated_joint_names]
        self.yourdf2mot = [self.urdf_obj.actuated_joint_names.index(jn) for jn in self.config["motor_joint"]]
        self.fixed_joint_ids = [i for i, jn in enumerate(self.urdf_obj.actuated_joint_names) if jn not in self.config["robot"]["movable_joints"]]

        # Robot
        self.robot = pk.Robot.from_urdf(self.urdf_obj)

        # Robot collision
        collision_cfg = self.config.get('collision', {})
        ignore_pairs = list(tuple(pair) for pair in collision_cfg.get('ignore_pairs', []))
        exclude_links = collision_cfg.get('exclude_links', [])
        ignore_pairs += list(product(exclude_links, exclude_links))
        self.robot_coll = RobotCollision.from_urdf(
            self.urdf_obj,
            user_ignore_pairs=ignore_pairs,
            ignore_immediate_adjacents=collision_cfg.get('ignore_adjacent_links', True),
        )

        coll_world = self.robot_coll.coll
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
        coll_world = jax.tree.map(lambda *args: jnp.stack(args), *cap_list)
        self.robot_coll = RobotCollision(
            self.robot_coll.num_links,
            self.robot_coll.link_names,
            coll_world,
            self.robot_coll.active_idx_i,
            self.robot_coll.active_idx_j,
        )

        self.root_link_name = "pelvis"
        self.target_link_name = "end_effector"
        self.target_link_idx = self.robot.links.names.index(self.target_link_name)
        self.len_traj = 100
        self.dt = 0.02


    def _load_obstacle(self):
        obstacle_urdf = yourdfpy.URDF.load(self.config["obstacle"]["urdf_path"])
        obstacle_coll_urdf = RobotCollision.from_urdf(obstacle_urdf)
        obstacle_coll = obstacle_coll_urdf.coll
        z_down_se3 = jaxlie.SE3.from_translation(np.array([0,0,-0.3]))
        obstacle_coll = obstacle_coll.transform(z_down_se3)
        obstacle_link_T = []
        for link_name in obstacle_coll_urdf.link_names:
            link_T = np.array(obstacle_urdf.get_transform(link_name))
            obstacle_link_T.append(link_T)
        obstacle_link_T = jnp.stack(obstacle_link_T)
        obstacle_link_se3 = jaxlie.SE3.from_matrix(obstacle_link_T)
        obstacle_coll = obstacle_coll.transform(obstacle_link_se3)

        return obstacle_coll


    def _load_welding_object_coll(self):
        N = 40
        radius = np.array(N*[0.0025])
        height = np.array(N*[0.15])
        pos = np.zeros((N,3))
        pos[:N//2,0] = np.linspace(-0.0475,0.0475,N//2)
        pos[:N//2,2] = 0.0025
        pos[N//2:,2] = np.linspace(0.0075,0.1025,N//2)
        wxyz = jaxlie.SO3.from_x_radians(np.array(N*[np.pi/2])).wxyz
        welding_object_coll = Capsule.from_radius_height(radius=radius, height=height, position=pos, wxyz=wxyz)

        return welding_object_coll


    def _load_collision_scene(self):
        obstacle_coll = self._load_obstacle()
        welding_object_coll = self._load_welding_object_coll()
        self.scene_coll = jax.tree.map(lambda x, y: jnp.concat([x, y]), obstacle_coll, welding_object_coll)


    def _get_current_state(self):
        # TODO: enable in real.
        # curr_joint_pos_mot = np.zeros(self.num_joints)
        # for i in range(self.num_joints):
        #     curr_joint_pos_mot[i] = self._low_state.motor_state[i].q
        # curr_joint_pos = curr_joint_pos_mot[self.mot2yourdf]
        # # Hard-codded temporally.
        # # curr_joint_pos = np.array([
        # #     -1.8665293 ,  0.1121762 , -0.04860201,  2.0799832 , -0.69911444,
        # #     -0.02357982, -1.7409432 ,  0.02690366,  0.06874572,  2.112137  ,
        # #     -0.85512745,  0.09089185,  0.01517932,  0.01537975, -0.01664756,
        # #     -0.21769302,  0.34277475,  0.01119563,  0.987899  , -0.00523058,
        # #      0.00483096,  0.01197389, -0.20725259, -0.33734336,  0.01429845,
        # #      0.98550737, -0.01143866,  0.01741067,  0.00395602
        # # ])

        # # TODO: implment this.
        # obj_pos_root, obj_quat_root = body_pose(self.tf_buffer, "welding_object", self.root_link_name)
        # # Hard-coded temporally.
        # # obj_pos_root, obj_quat_root = np.array([0.3974624 , -0.43468255,  0.03767955]), np.array([0.92344105,  0.08680602, -0.2230393 , -0.29995826])
        # welding_object_pose = jaxlie.SE3.from_rotation_and_translation(
        #     rotation=jaxlie.SO3(obj_quat_root),
        #     translation=obj_pos_root,
        # )

        curr_joint_pos = self.q
        welding_object_pose = self.welding_object_pose

        return curr_joint_pos, welding_object_pose
    

    def transform_se3_to_root(self, se3, frame_id):
        frame_pos_root, frame_quat_root = body_pose(
            self.tf_buffer,
            frame_id,
            self.root_link_name,
        )
        frame_pose_root = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(frame_quat_root),
            translation=frame_pos_root,
        )

        se3_root = frame_pose_root @ se3
        return se3_root


    def _init_sol_traj(self, curr_joint_pos, target_ee_pose_se3):
        curr_ee_se3 = jaxlie.SE3(self.robot.forward_kinematics(curr_joint_pos)[self.target_link_idx])
        to_welding_init_se3 = jaxlie.SE3.exp(
            jnp.linspace(
                curr_ee_se3.log(),
                target_ee_pose_se3.log(),
                self.len_traj,
            )
        )
        start_ee_pos = np.array(curr_ee_se3.translation())
        end_ee_pos = np.array(target_ee_pose_se3.translation())
        waypoints = np.array([
            start_ee_pos,
            (0.75*start_ee_pos + 0.25*end_ee_pos) + [-0.075, 0.0, 0.05],
            (0.25*start_ee_pos + 0.75*end_ee_pos) + [-0.075, 0.0, 0.05],
            end_ee_pos,
        ])
        waypoints = create_bspline_traj_from_points(waypoints, self.len_traj)
        to_welding_init_se3 = jaxlie.SE3.from_rotation_and_translation(
            to_welding_init_se3.rotation(),
            waypoints,
        )
        init_sol_traj = solve_ik_batch(
            self.robot,
            self.target_link_name,
            to_welding_init_se3.rotation().wxyz,
            to_welding_init_se3.translation(),
        )
        init_sol_traj[0] = curr_joint_pos
        init_sol_traj[:,:-7] = curr_joint_pos[:-7]

        return init_sol_traj, waypoints


    def _warmup_jit_fn(self):
        dummy_ee_poses_se3 = jaxlie.SE3.identity((self.len_traj,))
        init_sol_traj = solve_ik_batch(
            self.robot,
            self.target_link_name,
            dummy_ee_poses_se3.rotation().wxyz,
            dummy_ee_poses_se3.translation(),
        )

        dummy_ee_pose_se3 = jaxlie.SE3.identity()
        dummy_waypoints = np.zeros((self.len_traj, 3))
        pks.solve_online_planning(
            robot=self.robot,
            robot_coll=self.robot_coll,
            world_coll=self.scene_coll,
            target_link_name=self.target_link_name,
            target_position=dummy_ee_pose_se3.translation(),
            target_wxyz=dummy_ee_pose_se3.rotation().wxyz,
            timesteps=self.len_traj,
            dt=self.dt,
            start_cfg=init_sol_traj[0],
            prev_sols=init_sol_traj,
            fixed_joint_ids=self.fixed_joint_ids,
            default_joint_pos=np.zeros(self.num_joints),
            waypoints=dummy_waypoints,
        )


    def execute_trajopt_single_ee(self, goal_handle):
        self.get_logger().info('Executing goal...')

        curr_joint_pos, welding_object_pose = self._get_current_state()
        curr_scene_coll = self.scene_coll.transform(welding_object_pose)

        target_ee_pose_msg = goal_handle.request.ee_goal
        target_ee_pose_se3 = pose_stamped_msg_to_se3(target_ee_pose_msg)
        if target_ee_pose_msg.header.frame_id != self.root_link_name:
            target_ee_pose_se3 = self.transform_se3_to_root(target_ee_pose_se3, target_ee_pose_msg.header.frame_id)

        init_sol_traj, waypoints = self._init_sol_traj(curr_joint_pos, target_ee_pose_se3)
        sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(
            robot=self.robot,
            robot_coll=self.robot_coll,
            world_coll=curr_scene_coll,
            target_link_name=self.target_link_name,
            target_position=target_ee_pose_se3.translation(),
            target_wxyz=target_ee_pose_se3.rotation().wxyz,
            timesteps=self.len_traj,
            dt=self.dt,
            start_cfg=init_sol_traj[0],
            prev_sols=init_sol_traj,
            fixed_joint_ids=self.fixed_joint_ids,
            default_joint_pos=curr_joint_pos,
            waypoints=waypoints,
        )

        stamp = self.get_clock().now().to_msg()
        frame_id = self.root_link_name

        result = TrajOptSingleEE.Result()
        result.success = True
        result.joint_traj.header.stamp = stamp
        result.joint_traj.header.frame_id = frame_id

        result.joint_traj.joint_names = self.robot.joints.actuated_names

        joint_traj = sol_traj.tolist()
        for joint in joint_traj:
            point = JointTrajectoryPoint(positions=joint)
            result.joint_traj.points.append(point)
        goal_handle.succeed()
        self.get_logger().info(f'Succeed to find successful joint trajectories!!')

        self._set_vis_trajectory(welding_object_pose, sol_traj, sol_pos, sol_wxyz)

        return result


    def _add_collision_mesh(self):
        link_indices_for_collision = [self.robot.links.names.index(name) for name in self.robot_coll.link_names]
        # Update each collision capsule
        self.link_coll_vis = {}
        fk_poses_arr = self.robot.forward_kinematics(cfg=np.zeros(self.num_joints))
        fk_results_collision = fk_poses_arr[jnp.array(link_indices_for_collision)]
        coll_world = self.robot_coll.coll
        for i, link_name in enumerate(self.robot_coll.link_names):
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
            self.link_coll_vis[link_name] = self.server.scene.add_mesh_trimesh(
                f"collision/{link_name}",
                wireframe_mesh,
            )
            self.link_coll_vis[link_name].position = fk_results_collision[i, 4:]
            self.link_coll_vis[link_name].wxyz = fk_results_collision[i, :4]


    def _init_visualizer(self):
        self.server = viser.ViserServer()

        self.playing = self.server.gui.add_checkbox("playing", True)
        self.timestep_slider = self.server.gui.add_slider("timestep", 0, self.len_traj - 1, 1, 0)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.urdf_vis = ViserUrdf(self.server, self.urdf_obj, root_node_name="/robot")
        self.scene_coll_vis = self.server.scene.add_mesh_trimesh("/scene", mesh=self.scene_coll.to_trimesh())
        # welding_object_vis = server.scene.add_mesh_trimesh("/welding_object", mesh=welding_object_coll.to_trimesh())
        self.target_frame_handle = self.server.scene.add_batched_axes(
            "target_frame",
            axes_length=0.05,
            axes_radius=0.005,
            batched_positions=np.zeros((25, 3)),
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * 25),
        )
        self._add_collision_mesh()

        self.vis = False


    def _set_vis_trajectory(self, welding_object_pose, joint_traj, pos_traj, wxyz_traj):
        self.vis = True

        self.joint_traj_vis = joint_traj

        self.scene_coll_vis.position = welding_object_pose.translation()
        self.scene_coll_vis.wxyz = welding_object_pose.rotation().wxyz

        self.target_frame_handle.batched_positions = np.array(pos_traj)  # type: ignore[attr-defined]
        self.target_frame_handle.batched_wxyzs = np.array(wxyz_traj)  # type: ignore[attr-defined]


    def _vis_trajectory_callback(self):
        link_indices_for_collision = [self.robot.links.names.index(name) for name in self.robot_coll.link_names]
        while True:
            time.sleep(0.02)

            if not self.vis:
                continue

            with self.server.atomic():
                if self.playing.value:
                    self.timestep_slider.value = (self.timestep_slider.value + 1) % self.len_traj
                t = self.timestep_slider.value

                # Update visualizer.
                self.urdf_vis.update_cfg(
                    self.joint_traj_vis[t]
                )  # The first step of the online trajectory solution.

                # Update collision capsules
                fk_poses_arr = self.robot.forward_kinematics(cfg=self.joint_traj_vis[t])
                fk_results_collision = fk_poses_arr[jnp.array(link_indices_for_collision)]
                for i, link_name in enumerate(self.robot_coll.link_names):
                    self.link_coll_vis[link_name].position = fk_results_collision[i, 4:]
                    self.link_coll_vis[link_name].wxyz = fk_results_collision[i, :4]


def main(args=None):
    rclpy.init(args=args)
    trajopt_server = TrajOptSingleEEActionServer()
    try:
        rclpy.spin(trajopt_server)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()