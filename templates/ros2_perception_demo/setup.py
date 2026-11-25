from glob import glob
from setuptools import find_packages, setup
name = 'ros2_perception_demo'
setup(name=name, version='0.1.0', packages=find_packages(exclude=['tests']),
      data_files=[('share/ament_index/resource_index/packages', ['resource/'+name]),
                  ('share/'+name, ['package.xml']),
                  ('share/'+name+'/launch', glob('launch/*.launch.py')),
                  ('share/'+name+'/config', glob('config/*.yaml'))],
      install_requires=['setuptools', 'robotic-inference-sdk>=0.1,<0.2', 'numpy>=1.23'],
      maintainer='Application Maintainer', maintainer_email='maintainer@example.com',
      description='Independent perception application', license='Apache-2.0',
      entry_points={'console_scripts': ['perception_node = '+name+'.perception_node:main']})
