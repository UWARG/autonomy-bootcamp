"""
The real publisher, the one that actually sends ROS messages.

Two methods, ``publish_image`` and ``publish_status``. That's the whole
thing CaptureForPerception depends on, which is why a fake can stand in for
it in the tests.
"""

from __future__ import annotations

import json

import rclpy.node
from sensor_msgs.msg import Image
from std_msgs.msg import String

IMAGE_TOPIC = "/perception/image"
STATUS_TOPIC = "/perception/status"


class PerceptionPublisher:
    """Sends captured frames and status updates on two ROS topics."""

    def __init__(self, node: rclpy.node.Node) -> None:
        self._node = node
        self._image_pub = node.create_publisher(Image, IMAGE_TOPIC, 10)
        self._status_pub = node.create_publisher(String, STATUS_TOPIC, 10)
        # How many images we've sent so far. The mission reads this for the
        # "captures" field in mission_result.json.
        self.image_count = 0

    def publish_image(self, frame) -> None:
        """Send ``frame.rgb`` (H x W x 3 uint8) as a sensor_msgs/Image.

        Also counts the send, which is what the mission reports as
        ``captures`` and the Part 5 tests check.
        """
        height, width = frame.rgb.shape[:2]

        msg = Image()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = frame.rgb.tobytes()

        self._image_pub.publish(msg)
        self.image_count += 1

    def publish_status(self, status: dict) -> None:
        """Send ``status`` as JSON in a std_msgs/String.

        JSON in a string keeps this readable with ``ros2 topic echo`` without
        defining a custom message type.
        """
        self._status_pub.publish(String(data=json.dumps(status)))
