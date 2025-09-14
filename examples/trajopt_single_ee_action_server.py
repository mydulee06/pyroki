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
        self.get_logger().info("trajopt_single_ee_action_server starts!")

        self._load_config()
        self._load_robot()
        self._load_collision_scene()

        self._init_visualizer()
        threading.Thread(target=self._vis_trajectory_callback, daemon=True).start()

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


    def _low_state_cb(self, msg: LowState):
        self._low_state = msg


    def _load_config(self):
        self.asset_dir = Path(__file__).parent / "eetrack"
        config_file = self.asset_dir / "config_ros.yaml"
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)


    def _load_robot(self):
        urdf_path = self.config['robot']['urdf_path']
        self.urdf_obj = yourdfpy.URDF.load(urdf_path)
        self.num_joints = self.urdf_obj.num_dofs
        sit_terminal_states = np.load(self.config['robot']['sit_terminal_states_path'])
        idx = np.abs(sit_terminal_states["target_height"] - self.config['robot']['sit_target_height']).argmin()
        joint_pos = sit_terminal_states["joint_state"][idx, 0]
        lab2yourdf = [np.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in self.urdf_obj.actuated_joint_names]
        self.urdf_obj.update_cfg(joint_pos[lab2yourdf])

        self.mot2yourdf = [self.config["motor_joint"].index(jn) for jn in self.urdf_obj.actuated_joint_names]
        self.yourdf2mot = [self.urdf_obj.actuated_joint_names.index(jn) for jn in self.config["motor_joint"]]
        self.fixed_joint_ids = [i for i, jn in enumerate(self.urdf_obj.actuated_joint_names) if jn not in self.config["robot"]["movable_joints"]]

        # Robot
        self.robot = pk.Robot.from_urdf(self.urdf_obj)

        # Robot collision
        collision_cfg = self.config.get('collision', {})
        ignore_pairs = tuple(tuple(pair) for pair in collision_cfg.get('ignore_pairs', []))
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
        # Hard-codded temporally.
        curr_joint_pos = np.array([
            -2.0133293 ,  0.13313405, -0.05333192,  2.2461107 , -0.7422835 , -0.00471803, 
            -1.9364402 ,  0.07396649,  0.03822429,  2.2484412 , -0.81838095,  0.07413733,  
            0.00398889, -0.00195087, -0.04700555, -0.22352298,  0.33752224,  0.02540839,  
            1.0052238 ,  0.00622952, -0.01496937, -0.00291689, -0.23249201, -0.34823957,  
            0.01615466,  0.9942355 ,  0.00419126, -0.01379494,  0.00628898
        ])

        # TODO: implment this.
        # obj_pos_root, obj_quat_root = body_pose(self.tf_buffer, "welding_object", self.root_link_name, self.get_clock().now())
        # Hard-coded temporally.
        obj_pos_root, obj_quat_root = np.array([0.41655093, -0.41048726,  0.11140344]), np.array([0.9319723 ,  0.08422977, -0.23514102, -0.26275766])
        welding_object_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(obj_quat_root),
            translation=obj_pos_root,
        )

        return curr_joint_pos, welding_object_pose
    

    def transform_se3_to_root(self, se3, frame_id):
        frame_pos_root, frame_quat_root = body_pose(
            self.tf_buffer,
            frame_id,
            self.root_link_name,
            self.get_clock().now(),
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
        init_sol_traj = solve_ik_batch(
            self.robot,
            self.target_link_name,
            to_welding_init_se3.rotation().wxyz,
            to_welding_init_se3.translation(),
        )
        init_sol_traj[0] = curr_joint_pos
        init_sol_traj[:,:-7] = curr_joint_pos[:-7]

        return init_sol_traj


    def _warmup_jit_fn(self):
        dummy_ee_poses_se3 = jaxlie.SE3.identity((self.len_traj,))
        init_sol_traj = solve_ik_batch(
            self.robot,
            self.target_link_name,
            dummy_ee_poses_se3.rotation().wxyz,
            dummy_ee_poses_se3.translation(),
        )

        dummy_ee_pose_se3 = jaxlie.SE3.identity()
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
        )


    def execute_trajopt_single_ee(self, goal_handle):
        self.get_logger().info('Executing goal...')

        curr_joint_pos, welding_object_pose = self._get_current_state()
        curr_scene_coll = self.scene_coll.transform(welding_object_pose)

        target_ee_pose_msg = goal_handle.request.ee_goal
        target_ee_pose_se3 = pose_stamped_msg_to_se3(target_ee_pose_msg)
        if target_ee_pose_msg.header.frame_id != self.root_link_name:
            target_ee_pose_se3 = self.transform_se3_to_root(target_ee_pose_se3, target_ee_pose_msg.header.frame_id)

        init_sol_traj = self._init_sol_traj(curr_joint_pos, target_ee_pose_se3)
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
                f"collision_capsule_{link_name}",
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