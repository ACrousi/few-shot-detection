"""Scheme A: ROS launch starts an explicit venv interpreter; inference imports no ROS."""
from pathlib import Path
import sys
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    default = str(Path(get_package_share_directory('ros2_perception_demo'))/'config'/'default.yaml')
    package_path = str(Path(get_package_prefix('ros2_perception_demo'))/'lib'/
                       f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages')
    runner = ('import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); '
              'sys.argv[0]="ros2_perception_demo"; '
              'runpy.run_module("ros2_perception_demo",run_name="__main__",alter_sys=True)')
    return LaunchDescription([
        DeclareLaunchArgument('venv_python', description='Absolute path to application venv/bin/python'),
        DeclareLaunchArgument('config_path', default_value=default),
        ExecuteProcess(cmd=[LaunchConfiguration('venv_python'), '-I', '-c', runner, package_path,
                            '--config', LaunchConfiguration('config_path')], output='screen')])
