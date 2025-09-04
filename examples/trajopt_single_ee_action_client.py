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

from pyroki_ros.action import TrajOptSingleEE


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
            TrajOptSingleEE,
            'trajopt_single_ee',
        )


    def _load_config(self):
        self.asset_dir = Path(__file__).parent / "eetrack"
        config_file = self.asset_dir / "config_ros.yaml"
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)


    def send_ee_traj(self):
        self.get_logger().info('Start sending end effector trajectories...')

        stamp = self.get_clock().now().to_msg()
        frame_id = "pelvis"

        ee_goal_pos = np.array([0.43684632, -0.34309226,  0.13895929])
        ee_goal_wxyz = np.array([0.9510147 ,  0.17837143,  0.13940842, -0.21052338])

        goal_msg = TrajOptSingleEE.Goal()
        goal_msg.ee_goal.header.stamp = stamp
        goal_msg.ee_goal.header.frame_id = frame_id

        goal_msg.ee_goal.pose.position.x = ee_goal_pos[0]
        goal_msg.ee_goal.pose.position.y = ee_goal_pos[1]
        goal_msg.ee_goal.pose.position.z = ee_goal_pos[2]

        goal_msg.ee_goal.pose.orientation.w = ee_goal_wxyz[0]
        goal_msg.ee_goal.pose.orientation.x = ee_goal_wxyz[1]
        goal_msg.ee_goal.pose.orientation.y = ee_goal_wxyz[2]
        goal_msg.ee_goal.pose.orientation.z = ee_goal_wxyz[3]

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