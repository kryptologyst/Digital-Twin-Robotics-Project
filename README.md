# Digital Twin Robotics Project

**DISCLAIMER: This is a research/educational project. DO NOT use on real hardware without expert review and proper safety measures.**

## Overview

This project implements a comprehensive Digital Twin framework for robotics, providing real-time simulation, control, and optimization capabilities. The digital twin mirrors physical robot behavior in a virtual environment, enabling safe testing and development of control algorithms.

## Features

- **Real-time Digital Twin Simulation**: Physics-based simulation with sensor modeling
- **Multiple Robot Platforms**: Support for mobile robots, manipulators, and aerial vehicles
- **Advanced Control Systems**: PID, LQR, MPC, and learning-based controllers
- **Sensor Integration**: IMU, cameras, LiDAR, encoders with realistic noise models
- **ROS 2 Integration**: Full ROS 2 ecosystem support with launch files
- **Interactive Visualization**: Real-time plotting and 3D visualization
- **Evaluation Framework**: Comprehensive metrics and benchmarking tools

## Quick Start

### Prerequisites

- Python 3.10+
- ROS 2 Humble (optional, for ROS integration)
- CUDA-capable GPU (optional, for acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Digital-Twin-Robotics-Project.git
cd Digital-Twin-Robotics-Project

# Install in development mode
pip install -e ".[dev,simulation,learning]"

# For ROS 2 integration
pip install -e ".[ros2]"
```

### Basic Usage

```python
from src.digital_twin import DigitalTwinRobot
from src.controllers import PIDController
from src.sensors import IMUSensor, CameraSensor

# Create a digital twin robot
robot = DigitalTwinRobot(
    robot_type="mobile",
    initial_pose=[0.0, 0.0, 0.0],
    physics_engine="pybullet"
)

# Add sensors
robot.add_sensor(IMUSensor(noise_std=0.01))
robot.add_sensor(CameraSensor(resolution=(640, 480)))

# Set up controller
controller = PIDController(kp=1.0, ki=0.1, kd=0.05)

# Run simulation
for step in range(1000):
    # Get sensor data
    sensor_data = robot.get_sensor_data()
    
    # Compute control action
    action = controller.compute_action(sensor_data, target_pose=[5.0, 5.0, 0.0])
    
    # Apply action to robot
    robot.step(action)
    
    # Update digital twin state
    robot.update_digital_twin()
```

### ROS 2 Launch

```bash
# Launch the digital twin with ROS 2
ros2 launch digital_twin_robotics digital_twin.launch.py

# Launch with specific robot configuration
ros2 launch digital_twin_robotics mobile_robot.launch.py robot_type:=differential_drive
```

## Project Structure

```
src/
├── digital_twin/          # Core digital twin implementation
├── controllers/           # Control algorithms (PID, LQR, MPC, RL)
├── sensors/              # Sensor models and data processing
├── robots/               # Robot platform implementations
├── physics/              # Physics simulation engines
├── visualization/        # Plotting and 3D visualization
├── evaluation/           # Metrics and benchmarking
└── utils/                # Utility functions and helpers

robots/
├── urdf/                 # Robot descriptions
├── meshes/              # 3D models
└── config/              # Robot-specific configurations

launch/                  # ROS 2 launch files
config/                  # YAML configuration files
data/                    # Datasets and recorded data
scripts/                 # Utility scripts
notebooks/               # Jupyter notebooks for analysis
tests/                   # Unit and integration tests
assets/                  # Generated plots, videos, and demos
demo/                    # Interactive demos
```

## Robot Platforms

### Mobile Robots
- **Differential Drive**: Two-wheeled mobile robots
- **Ackermann**: Car-like vehicles with steering
- **Omnidirectional**: Holonomic mobile platforms

### Manipulators
- **6-DOF Arms**: Standard industrial manipulators
- **7-DOF Arms**: Redundant manipulators
- **Parallel Robots**: Delta and Stewart platforms

### Aerial Vehicles
- **Quadrotors**: Standard multirotor UAVs
- **Fixed-wing**: Aircraft with aerodynamic models

## Control Systems

### Classical Control
- **PID Controllers**: Proportional-Integral-Derivative control
- **LQR/LQG**: Linear Quadratic Regulator/Gaussian
- **MPC**: Model Predictive Control with constraints

### Advanced Control
- **Adaptive Control**: Self-tuning controllers
- **Robust Control**: H-infinity and sliding mode
- **Nonlinear Control**: Feedback linearization

### Learning-based Control
- **Reinforcement Learning**: PPO, SAC, TD3
- **Imitation Learning**: Behavior cloning, GAIL
- **Neural Networks**: Deep learning controllers

## Sensors

### Inertial Sensors
- **IMU**: Accelerometer, gyroscope, magnetometer
- **Encoders**: Wheel and joint encoders

### Vision Sensors
- **Cameras**: RGB, depth, stereo cameras
- **LiDAR**: 2D and 3D laser scanners

### Environmental Sensors
- **GPS**: Global positioning system
- **Odometry**: Wheel and visual odometry

## Evaluation Metrics

### Control Performance
- **Tracking Error**: RMSE of position/orientation
- **Settling Time**: Time to reach steady state
- **Overshoot**: Maximum deviation from target
- **Control Effort**: Energy consumption

### Navigation Performance
- **Path Length**: Distance traveled vs optimal
- **Success Rate**: Percentage of successful missions
- **Collision Rate**: Safety metrics

### Learning Performance
- **Sample Efficiency**: Episodes to convergence
- **Final Performance**: Average return
- **Stability**: Performance variance

## Safety Features

- **Velocity Limits**: Maximum speed constraints
- **Effort Limits**: Torque/force constraints
- **Emergency Stop**: Immediate halt capability
- **Collision Detection**: Real-time safety monitoring
- **Dry-run Mode**: Simulation-only operation

## Configuration

The system uses Hydra for configuration management:

```yaml
# config/default.yaml
robot:
  type: mobile
  platform: differential_drive
  
physics:
  engine: pybullet
  timestep: 0.01
  gravity: [0, 0, -9.81]
  
sensors:
  imu:
    enabled: true
    noise_std: 0.01
  camera:
    enabled: true
    resolution: [640, 480]
    
controller:
  type: pid
  gains:
    kp: 1.0
    ki: 0.1
    kd: 0.05
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{digital_twin_robotics,
  title={Digital Twin for Robotics: A Comprehensive Simulation Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Digital-Twin-Robotics-Project}
}
```
# Digital-Twin-Robotics-Project
