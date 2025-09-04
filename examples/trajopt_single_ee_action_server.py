import os
import sys
import time
import yaml
import yourdfpy
import numpy as np
from pathlib import Path
from typing import TypedDict
import threading

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
        urdf_obj = yourdfpy.URDF.load(urdf_path)
        self.num_joints = urdf_obj.num_dofs
        sit_terminal_states = np.load(self.config['robot']['sit_terminal_states_path'])
        idx = np.abs(sit_terminal_states["target_height"] - self.config['robot']['sit_target_height']).argmin()
        joint_pos = sit_terminal_states["joint_state"][idx, 0]
        lab2yourdf = [np.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in urdf_obj.actuated_joint_names]
        urdf_obj.update_cfg(joint_pos[lab2yourdf])
        for joint in urdf_obj.robot.joints:
            if joint.name in urdf_obj.actuated_joint_names and joint.name not in self.config['robot']['movable_joints']:
                joint.type = "fixed"
                joint.origin = urdf_obj.get_transform(joint.child, joint.parent)
        self.modified_urdf = yourdfpy.URDF(urdf_obj.robot, mesh_dir=Path(urdf_path).parent)

        self.mot2yourdf = [self.config["motor_joint"].index(jn) for jn in urdf_obj.actuated_joint_names]
        self.yourdf2mot = [urdf_obj.actuated_joint_names.index(jn) for jn in self.config["motor_joint"]]

        # Robot
        self.robot = pk.Robot.from_urdf(self.modified_urdf)

        # Robot collision
        collision_cfg = self.config.get('collision', {})
        ignore_pairs = tuple(tuple(pair) for pair in collision_cfg.get('ignore_pairs', []))
        self.robot_coll = RobotCollision.from_urdf(
            self.modified_urdf,
            user_ignore_pairs=ignore_pairs,
            ignore_immediate_adjacents=collision_cfg.get('ignore_adjacent_links', True),
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
        curr_right_arm_joint_pos = curr_joint_pos[-7:]

        # TODO: implment this.
        # obj_pos_root, obj_quat_root = body_pose(self.tf_buffer, "welding_object", self.root_link_name, self.get_clock().now())
        # Hard-coded temporally.
        obj_pos_root, obj_quat_root = np.array([0.41655093, -0.41048726,  0.11140344]), np.array([0.9319723 ,  0.08422977, -0.23514102, -0.26275766])
        welding_object_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(obj_quat_root),
            translation=obj_pos_root,
        )

        return curr_right_arm_joint_pos, welding_object_pose
    

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


    def _init_visualizer(self):
        self.server = viser.ViserServer()

        self.playing = self.server.gui.add_checkbox("playing", True)
        self.timestep_slider = self.server.gui.add_slider("timestep", 0, self.len_traj - 1, 1, 0)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.urdf_vis = ViserUrdf(self.server, self.modified_urdf, root_node_name="/robot")
        self.scene_coll_vis = self.server.scene.add_mesh_trimesh("/scene", mesh=self.scene_coll.to_trimesh())
        # welding_object_vis = server.scene.add_mesh_trimesh("/welding_object", mesh=welding_object_coll.to_trimesh())
        self.target_frame_handle = self.server.scene.add_batched_axes(
            "target_frame",
            axes_length=0.05,
            axes_radius=0.005,
            batched_positions=np.zeros((25, 3)),
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * 25),
        )

        self.vis = False


    def _set_vis_trajectory(self, welding_object_pose, joint_traj, pos_traj, wxyz_traj):
        self.vis = True

        self.joint_traj_vis = joint_traj

        self.scene_coll_vis.position = welding_object_pose.translation()
        self.scene_coll_vis.wxyz = welding_object_pose.rotation().wxyz

        self.target_frame_handle.batched_positions = np.array(pos_traj)  # type: ignore[attr-defined]
        self.target_frame_handle.batched_wxyzs = np.array(wxyz_traj)  # type: ignore[attr-defined]


    def _vis_trajectory_callback(self):
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