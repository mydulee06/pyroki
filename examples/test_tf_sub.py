import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


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


class TestTFSubscriber(Node):
    def __init__(self):
        super().__init__('tf_sub_tester')
        self.get_logger().info("TF sub test starts!")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.create_timer(1.0, self.tf_sub_cb)


    def tf_sub_cb(self):
        body_pose(
            self.tf_buffer,
            "welding_object",
            "world",
        )
        print(self.get_clock().now())
        print(rclpy.time.Time())


def main(args=None):
    rclpy.init(args=args)
    tf_sub = TestTFSubscriber()
    try:
        rclpy.spin(tf_sub)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()