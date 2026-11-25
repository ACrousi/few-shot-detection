# Platform-owned application template

The default configuration executes `platforms/cpu/pipeline.py`, a complete mean-RGB
MOCK example with no AI inference. Install the package, then run:

```bash
python -m ros2_perception_demo --config config/default.yaml
rsctl validate --config config/default.yaml
```

Vendor files under `platforms/nvidia`, `platforms/intel`, `platforms/qualcomm` are explicit
not-implemented scaffolds. Add the whole runtime, preprocessing, transfer and postprocessing
there. Update that platform's YAML. There is no model adapter or backend interface.

The ROS console entry runs the CPU example's `platforms/cpu/ros_node.py`; real platforms
supply their own ROS integration. ROS requires compatible rclpy/cv_bridge dependencies.
