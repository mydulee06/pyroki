import os
from pathlib import Path
import yaml
import numpy as np
import yourdfpy

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Pyroki
import jax.numpy as jnp
import jaxlie
from eetrack.utils.weld_objects import WeldObject

# ROS
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped

from pyroki_ros.action import TrajOpt


def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + right_sole_link_pose.rotation().log()) / 2
        ),
        translation=(left_sole_link_pose.translation() + right_sole_link_pose.translation()) / 2,
    )


class TrajOptActionClient(Node):
    def __init__(self):
        super().__init__('trajopt_action_client')
        self.get_logger().info("trajopt_action_client starts!")

        # Initialization
        self._load_config()

        self._action_client = ActionClient(
            self,
            TrajOpt,
            'trajopt',
        )


    def _load_config(self):
        self.asset_dir = Path(__file__).parent / "eetrack"
        config_file = self.asset_dir / "config_ros.yaml"
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)


    def _sample_welding_path(self):
        # Sample mid_sole_pose
        search_space = self.config.get('search_space', {})
        x_min, x_max = search_space.get('x_range', [-0.3, 0.3])
        y_min, y_max = search_space.get('y_range', [-0.5, -0.1])
        yaw_min, yaw_max = search_space.get('angle_range', [-np.pi/2, np.pi/2])
        self.x_mid_sole_obj = np.random.uniform(x_min, x_max)
        self.y_mid_sole_obj = np.random.uniform(y_min, y_max)
        self.z_mid_sole_obj = -search_space.get('z_height', 0.0)
        self.yaw_mid_sole_obj = np.random.uniform(yaw_min, yaw_max)

        # Load urdf
        urdf_path = self.config['robot']['urdf_path']
        urdf_obj = yourdfpy.URDF.load(urdf_path)
        sit_terminal_states = np.load(self.config['robot']['sit_terminal_states_path'])
        idx = np.abs(sit_terminal_states["target_height"] - self.config['robot']['sit_target_height']).argmin()
        joint_pos = sit_terminal_states["joint_state"][idx, 0]
        lab2yourdf = [np.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in urdf_obj.actuated_joint_names]
        urdf_obj.update_cfg(joint_pos[lab2yourdf])
        for joint in urdf_obj.robot.joints:
            if joint.name in urdf_obj.actuated_joint_names and joint.name not in self.config['robot']['movable_joints']:
                joint.type = "fixed"
                joint.origin = urdf_obj.get_transform(joint.child, joint.parent)
        modified_urdf = yourdfpy.URDF(urdf_obj.robot, mesh_dir=Path(urdf_path).parent)

        # Init welding object
        welding_object_config = self.config["welding_object"].copy()
        welding_object_config.pop('pose', None)
        welding_object_config.pop('yaw', None)
        welding_object_config.pop("parent", None)

        left_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("left_sole_link")[None])
        right_sole = jaxlie.SE3.from_matrix(modified_urdf.get_transform("right_sole_link")[None])
        mid_sole_pose_pelvis = get_mid_sole_link_pose(left_sole, right_sole)
        mid_sole_pos_obj = jnp.array([
            self.x_mid_sole_obj,
            self.y_mid_sole_obj,
            self.z_mid_sole_obj,
        ])
        mid_sole_so3_obj = jaxlie.SO3.from_rpy_radians(0.0, 0.0, self.yaw_mid_sole_obj)
        mid_sole_pose_obj = jaxlie.SE3.from_rotation_and_translation(mid_sole_so3_obj, mid_sole_pos_obj)

        welding_object_pose = mid_sole_pose_pelvis @ mid_sole_pose_obj.inverse()
        welding_object = WeldObject(**welding_object_config)

        # Get welding path
        welding_path_se3 = welding_object.get_welding_path(welding_object_pose)
        welding_path_pos = welding_path_se3.translation()
        welding_path_wxyz = welding_path_se3.rotation().wxyz
        welding_xyz_wxyz_path = jnp.concatenate([welding_path_pos, welding_path_wxyz], axis=-1)[0].tolist()

        return welding_xyz_wxyz_path


    def send_ee_traj(self):
        self.get_logger().info('Start sending end effector trajectories...')

        stamp = self.get_clock().now().to_msg()
        frame_id = "world"

        goal_msg = TrajOpt.Goal()
        goal_msg.ee_traj.header.stamp = stamp
        goal_msg.ee_traj.header.frame_id = frame_id

        welding_xyz_wxyz_path = self._sample_welding_path()

        for xyz_wxyz in welding_xyz_wxyz_path:
            pose_stamped = PoseStamped()

            pose_stamped.header.stamp = stamp
            pose_stamped.header.frame_id = frame_id

            pose_stamped.pose.position.x = xyz_wxyz[0]
            pose_stamped.pose.position.y = xyz_wxyz[1]
            pose_stamped.pose.position.z = xyz_wxyz[2]

            pose_stamped.pose.orientation.w = xyz_wxyz[3]
            pose_stamped.pose.orientation.x = xyz_wxyz[4]
            pose_stamped.pose.orientation.y = xyz_wxyz[5]
            pose_stamped.pose.orientation.z = xyz_wxyz[6]

            goal_msg.ee_traj.poses.append(pose_stamped)

        # Send goal to action server
        self.get_logger().info("Waiting for action server...")
        self._action_client.wait_for_server()

        self.get_logger().info("Sending goal end effector trajectories...")
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()

        self._get_result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        result = future.result().result

        joint_traj = result.joint_traj
        success = result.success
        error_message = result.error_message

        self.get_logger().info(f'Success: {success}')


    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        percent = feedback_msg.percent_complete
        self.get_logger().info(f'In progress: {percent:.1f}')


def main(args=None):
    rclpy.init(args=args)
    trajopt_client = TrajOptActionClient()

    trajopt_client.send_ee_traj()

    rclpy.spin(trajopt_client)


if __name__ == '__main__':
    main()