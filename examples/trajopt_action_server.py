import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from pyroki_ros.action import TrajOpt


class TrajOptActionServer(Node):
    def __init__(self):
        super().__init__('trajopt_action_server')
        self.get_logger().info("trajopt_action_server starts!")
        self._action_server = ActionServer(
            self,
            TrajOpt,
            'trajopt',
            self.execute_trajopt)


    def execute_trajopt(self, goal_handle):
        self.get_logger().info('Executing goal...')
        ee_traj = goal_handle.request.ee_traj
        # TODO: Traj opt with ee_traj.
        feedback_msg = TrajOpt.Feedback()
        feedback_msg.percent_complete = 0.5
        goal_handle.publish_feedback(feedback_msg)
        if success:
            result = TrajOpt.Result()
            result.joint_traj = None
            result.success = True
        else:
            result.success = False
            result.error_message = "TODO"
        return result


def main():
    rclpy.init()
    node = TrajOptActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()