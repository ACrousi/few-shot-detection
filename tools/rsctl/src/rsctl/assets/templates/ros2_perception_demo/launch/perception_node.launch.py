from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    default = str(Path(get_package_share_directory('ros2_perception_demo'))/'config'/'default.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config_path', default_value=default),
        Node(package='ros2_perception_demo', executable='perception_node',
             parameters=[{'config_path': LaunchConfiguration('config_path')}], output='screen')])
