import json
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from .pipeline import Pipeline

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_demo')
        self.declare_parameter('config_path', '')
        self.pipeline = Pipeline(self.get_parameter('config_path').value)
        ros = self.pipeline.settings.get('ros', {})
        self.declare_parameter('input_topic', ros.get('input_topic','/camera/color/image_raw'))
        self.declare_parameter('output_topic', ros.get('output_topic','/perception/mock_summary'))
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(String, self.get_parameter('output_topic').value, 10)
        self.subscription = self.create_subscription(Image, self.get_parameter('input_topic').value,
                                                     self.on_image, qos_profile_sensor_data)
        self.get_logger().warning('MOCK perception only: no medical recognition is performed')

    def on_image(self, msg):
        try:
            result = self.pipeline.process(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'))
            result['frame_id'] = msg.header.frame_id
            result['stamp'] = {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec}
            self.publisher.publish(String(data=json.dumps(result)))
        except Exception as e:
            self.get_logger().error(f'Frame failed: {e}')

    def destroy_node(self):
        self.pipeline.close()
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
