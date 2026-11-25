"""Run with system ROS Python after sourcing a generated package's install/setup.bash."""
import argparse
import importlib
import json
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

parser=argparse.ArgumentParser()
parser.add_argument('--package',required=True)
parser.add_argument('--config',required=True)
args=parser.parse_args()
rclpy.init(args=['--ros-args','-p',f'config_path:={args.config}'])
node=importlib.import_module(args.package+'.perception_node').PerceptionNode()
probe=rclpy.create_node('robotic_suite_test_probe')
received=[]
pub=probe.create_publisher(Image,'/camera/color/image_raw',qos_profile_sensor_data)
sub=probe.create_subscription(String,'/perception/mock_summary',lambda msg: received.append(json.loads(msg.data)),10)
executor=SingleThreadedExecutor()
executor.add_node(node); executor.add_node(probe)
try:
    message=Image(height=8,width=8,encoding='rgb8',step=24,data=np.full((8,8,3),128,dtype=np.uint8).tobytes())
    message.header.frame_id='test_camera'
    deadline=time.monotonic()+10
    while not received and time.monotonic()<deadline:
        pub.publish(message)
        executor.spin_once(timeout_sec=0.1)
    assert received, 'No ROS output within 10 seconds'
    assert received[0]['mode']=='MOCK'
    assert received[0]['frame_id']=='test_camera'
    np.testing.assert_allclose(received[0]['mean_rgb'],[128/255]*3,rtol=1e-5)
    print('ROS Jazzy image -> mock pipeline -> topic round trip PASSED')
finally:
    executor.shutdown()
    node.destroy_node(); probe.destroy_node(); rclpy.shutdown()
