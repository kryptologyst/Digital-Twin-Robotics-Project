"""
Core digital twin implementation for robotics.

This module provides the main DigitalTwinRobot class that orchestrates
simulation, control, and sensor integration for various robot platforms.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf

from ..controllers.base import BaseController
from ..physics.base import PhysicsEngine
from ..robots.base import BaseRobot
from ..sensors.base import BaseSensor
from ..utils.seeding import set_deterministic_seeds
from ..utils.device import get_device
from ..visualization.plotter import Plotter

logger = logging.getLogger(__name__)


class DigitalTwinRobot:
    """
    Main digital twin class for robotics simulation and control.
    
    This class orchestrates the integration of robot models, physics simulation,
    sensors, controllers, and visualization to create a comprehensive digital
    twin system.
    
    Args:
        robot_type: Type of robot ('mobile', 'manipulator', 'aerial')
        platform: Specific platform ('differential_drive', '6dof_arm', 'quadrotor')
        initial_pose: Initial robot pose [x, y, z, roll, pitch, yaw]
        physics_engine: Physics simulation engine ('pybullet', 'mujoco', 'gazebo')
        config: Configuration dictionary or path to config file
        device: Computing device ('cuda', 'mps', 'cpu')
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        robot_type: str = "mobile",
        platform: str = "differential_drive",
        initial_pose: Optional[List[float]] = None,
        physics_engine: str = "pybullet",
        config: Optional[Union[Dict[str, Any], str]] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        # Set deterministic seeds
        if seed is not None:
            set_deterministic_seeds(seed)
        
        # Device management
        self.device = get_device(device)
        
        # Configuration
        self.config = self._load_config(config)
        
        # Robot type and platform
        self.robot_type = robot_type
        self.platform = platform
        
        # Initial pose
        self.initial_pose = initial_pose or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Components
        self.robot: Optional[BaseRobot] = None
        self.physics_engine: Optional[PhysicsEngine] = None
        self.sensors: List[BaseSensor] = []
        self.controller: Optional[BaseController] = None
        self.plotter: Optional[Plotter] = None
        
        # State tracking
        self.current_pose = np.array(self.initial_pose, dtype=np.float32)
        self.current_velocity = np.zeros(6, dtype=np.float32)
        self.current_acceleration = np.zeros(6, dtype=np.float32)
        self.sensor_data: Dict[str, Any] = {}
        self.control_action: Optional[npt.NDArray[np.float32]] = None
        
        # History for analysis
        self.pose_history: List[npt.NDArray[np.float32]] = []
        self.velocity_history: List[npt.NDArray[np.float32]] = []
        self.action_history: List[npt.NDArray[np.float32]] = []
        self.sensor_history: List[Dict[str, Any]] = []
        
        # Simulation state
        self.simulation_time = 0.0
        self.timestep = self.config.get("timestep", 0.01)
        self.is_running = False
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Digital twin initialized: {robot_type}/{platform}")
    
    def _load_config(self, config: Optional[Union[Dict[str, Any], str]]) -> DictConfig:
        """Load configuration from file or dictionary."""
        if config is None:
            # Load default config
            default_config = {
                "timestep": 0.01,
                "physics": {
                    "engine": "pybullet",
                    "gravity": [0, 0, -9.81],
                    "real_time": False,
                },
                "robot": {
                    "type": self.robot_type,
                    "platform": self.platform,
                },
                "sensors": {
                    "imu": {"enabled": True, "noise_std": 0.01},
                    "camera": {"enabled": False, "resolution": [640, 480]},
                    "lidar": {"enabled": False, "range_max": 10.0},
                },
                "controller": {
                    "type": "pid",
                    "gains": {"kp": 1.0, "ki": 0.1, "kd": 0.05},
                },
                "visualization": {
                    "enabled": True,
                    "real_time": True,
                    "save_plots": False,
                },
            }
            return OmegaConf.create(default_config)
        
        if isinstance(config, str):
            return OmegaConf.load(config)
        
        return OmegaConf.create(config)
    
    def _initialize_components(self) -> None:
        """Initialize all digital twin components."""
        try:
            # Initialize physics engine
            self._initialize_physics()
            
            # Initialize robot
            self._initialize_robot()
            
            # Initialize sensors
            self._initialize_sensors()
            
            # Initialize controller
            self._initialize_controller()
            
            # Initialize visualization
            self._initialize_visualization()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_physics(self) -> None:
        """Initialize physics simulation engine."""
        engine_name = self.config.physics.engine.lower()
        
        if engine_name == "pybullet":
            from ..physics.pybullet_engine import PyBulletEngine
            self.physics_engine = PyBulletEngine(
                timestep=self.timestep,
                gravity=self.config.physics.gravity,
                real_time=self.config.physics.get("real_time", False),
            )
        elif engine_name == "mujoco":
            from ..physics.mujoco_engine import MuJoCoEngine
            self.physics_engine = MuJoCoEngine(
                timestep=self.timestep,
                gravity=self.config.physics.gravity,
            )
        else:
            raise ValueError(f"Unsupported physics engine: {engine_name}")
    
    def _initialize_robot(self) -> None:
        """Initialize robot model."""
        robot_type = self.config.robot.type.lower()
        platform = self.config.robot.platform.lower()
        
        if robot_type == "mobile":
            if platform == "differential_drive":
                from ..robots.mobile import DifferentialDriveRobot
                self.robot = DifferentialDriveRobot(
                    initial_pose=self.current_pose,
                    physics_engine=self.physics_engine,
                )
            elif platform == "ackermann":
                from ..robots.mobile import AckermannRobot
                self.robot = AckermannRobot(
                    initial_pose=self.current_pose,
                    physics_engine=self.physics_engine,
                )
            else:
                raise ValueError(f"Unsupported mobile platform: {platform}")
        
        elif robot_type == "manipulator":
            if platform == "6dof_arm":
                from ..robots.manipulator import SixDOFArm
                self.robot = SixDOFArm(
                    initial_pose=self.current_pose,
                    physics_engine=self.physics_engine,
                )
            else:
                raise ValueError(f"Unsupported manipulator platform: {platform}")
        
        elif robot_type == "aerial":
            if platform == "quadrotor":
                from ..robots.aerial import QuadrotorRobot
                self.robot = QuadrotorRobot(
                    initial_pose=self.current_pose,
                    physics_engine=self.physics_engine,
                )
            else:
                raise ValueError(f"Unsupported aerial platform: {platform}")
        
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
    
    def _initialize_sensors(self) -> None:
        """Initialize sensor suite."""
        sensor_config = self.config.sensors
        
        if sensor_config.imu.enabled:
            from ..sensors.imu import IMUSensor
            imu = IMUSensor(
                noise_std=sensor_config.imu.noise_std,
                device=self.device,
            )
            self.add_sensor(imu)
        
        if sensor_config.camera.enabled:
            from ..sensors.camera import CameraSensor
            camera = CameraSensor(
                resolution=sensor_config.camera.resolution,
                device=self.device,
            )
            self.add_sensor(camera)
        
        if sensor_config.lidar.enabled:
            from ..sensors.lidar import LiDARSensor
            lidar = LiDARSensor(
                range_max=sensor_config.lidar.range_max,
                device=self.device,
            )
            self.add_sensor(lidar)
    
    def _initialize_controller(self) -> None:
        """Initialize control system."""
        controller_type = self.config.controller.type.lower()
        gains = self.config.controller.gains
        
        if controller_type == "pid":
            from ..controllers.pid import PIDController
            self.controller = PIDController(
                kp=gains.kp,
                ki=gains.ki,
                kd=gains.kd,
                device=self.device,
            )
        elif controller_type == "lqr":
            from ..controllers.lqr import LQRController
            self.controller = LQRController(device=self.device)
        elif controller_type == "mpc":
            from ..controllers.mpc import MPCController
            self.controller = MPCController(device=self.device)
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")
    
    def _initialize_visualization(self) -> None:
        """Initialize visualization system."""
        if self.config.visualization.enabled:
            from ..visualization.plotter import Plotter
            self.plotter = Plotter(
                real_time=self.config.visualization.real_time,
                save_plots=self.config.visualization.save_plots,
            )
    
    def add_sensor(self, sensor: BaseSensor) -> None:
        """Add a sensor to the digital twin."""
        self.sensors.append(sensor)
        logger.info(f"Added sensor: {sensor.__class__.__name__}")
    
    def set_controller(self, controller: BaseController) -> None:
        """Set the control system."""
        self.controller = controller
        logger.info(f"Set controller: {controller.__class__.__name__}")
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """Get sensor data from all sensors."""
        sensor_data = {}
        
        for sensor in self.sensors:
            try:
                data = sensor.get_measurement(
                    robot_pose=self.current_pose,
                    robot_velocity=self.current_velocity,
                    simulation_time=self.simulation_time,
                )
                sensor_data[sensor.name] = data
            except Exception as e:
                logger.warning(f"Sensor {sensor.name} failed: {e}")
                sensor_data[sensor.name] = None
        
        self.sensor_data = sensor_data
        return sensor_data
    
    def compute_control_action(
        self,
        target_pose: Optional[npt.NDArray[np.float32]] = None,
        target_velocity: Optional[npt.NDArray[np.float32]] = None,
    ) -> Optional[npt.NDArray[np.float32]]:
        """Compute control action using the current controller."""
        if self.controller is None:
            logger.warning("No controller set")
            return None
        
        try:
            action = self.controller.compute_action(
                current_pose=self.current_pose,
                current_velocity=self.current_velocity,
                target_pose=target_pose,
                target_velocity=target_velocity,
                sensor_data=self.sensor_data,
            )
            self.control_action = action
            return action
        except Exception as e:
            logger.error(f"Control computation failed: {e}")
            return None
    
    def step(self, action: Optional[npt.NDArray[np.float32]] = None) -> None:
        """Execute one simulation step."""
        if action is None:
            action = self.control_action
        
        if action is None:
            logger.warning("No action provided for step")
            return
        
        # Apply action to robot
        if self.robot is not None:
            self.robot.apply_action(action)
        
        # Step physics simulation
        if self.physics_engine is not None:
            self.physics_engine.step()
        
        # Update robot state
        if self.robot is not None:
            self.current_pose = self.robot.get_pose()
            self.current_velocity = self.robot.get_velocity()
            self.current_acceleration = self.robot.get_acceleration()
        
        # Update simulation time
        self.simulation_time += self.timestep
        
        # Record history
        self.pose_history.append(self.current_pose.copy())
        self.velocity_history.append(self.current_velocity.copy())
        if action is not None:
            self.action_history.append(action.copy())
        self.sensor_history.append(self.sensor_data.copy())
    
    def update_digital_twin(self) -> None:
        """Update the digital twin state."""
        # Get sensor data
        self.get_sensor_data()
        
        # Compute control action
        self.compute_control_action()
        
        # Execute simulation step
        self.step()
        
        # Update visualization
        if self.plotter is not None:
            self.plotter.update(
                pose=self.current_pose,
                velocity=self.current_velocity,
                action=self.control_action,
                sensor_data=self.sensor_data,
                simulation_time=self.simulation_time,
            )
    
    def run_simulation(
        self,
        duration: float,
        target_pose: Optional[npt.NDArray[np.float32]] = None,
        target_velocity: Optional[npt.NDArray[np.float32]] = None,
    ) -> None:
        """Run simulation for specified duration."""
        self.is_running = True
        start_time = time.time()
        
        logger.info(f"Starting simulation for {duration} seconds")
        
        try:
            while self.simulation_time < duration and self.is_running:
                # Update digital twin
                self.update_digital_twin()
                
                # Compute control action with targets
                if target_pose is not None or target_velocity is not None:
                    self.compute_control_action(target_pose, target_velocity)
                
                # Real-time simulation pacing
                if self.config.physics.get("real_time", False):
                    elapsed = time.time() - start_time
                    if elapsed < self.simulation_time:
                        time.sleep(self.simulation_time - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        
        finally:
            self.is_running = False
            logger.info(f"Simulation completed. Final time: {self.simulation_time:.2f}s")
    
    def stop_simulation(self) -> None:
        """Stop the running simulation."""
        self.is_running = False
        logger.info("Simulation stopped")
    
    def reset(self) -> None:
        """Reset the digital twin to initial state."""
        self.current_pose = np.array(self.initial_pose, dtype=np.float32)
        self.current_velocity = np.zeros(6, dtype=np.float32)
        self.current_acceleration = np.zeros(6, dtype=np.float32)
        self.simulation_time = 0.0
        
        # Clear history
        self.pose_history.clear()
        self.velocity_history.clear()
        self.action_history.clear()
        self.sensor_history.clear()
        
        # Reset components
        if self.robot is not None:
            self.robot.reset()
        
        if self.physics_engine is not None:
            self.physics_engine.reset()
        
        if self.controller is not None:
            self.controller.reset()
        
        logger.info("Digital twin reset to initial state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current digital twin state."""
        return {
            "pose": self.current_pose.copy(),
            "velocity": self.current_velocity.copy(),
            "acceleration": self.current_acceleration.copy(),
            "simulation_time": self.simulation_time,
            "sensor_data": self.sensor_data.copy(),
            "control_action": self.control_action.copy() if self.control_action is not None else None,
        }
    
    def save_data(self, filepath: str) -> None:
        """Save simulation data to file."""
        data = {
            "pose_history": np.array(self.pose_history),
            "velocity_history": np.array(self.velocity_history),
            "action_history": np.array(self.action_history),
            "sensor_history": self.sensor_history,
            "config": OmegaConf.to_yaml(self.config),
            "simulation_time": self.simulation_time,
        }
        
        np.savez(filepath, **data)
        logger.info(f"Data saved to {filepath}")
    
    def close(self) -> None:
        """Clean up resources."""
        if self.physics_engine is not None:
            self.physics_engine.close()
        
        if self.plotter is not None:
            self.plotter.close()
        
        logger.info("Digital twin closed")
    
    def __enter__(self) -> DigitalTwinRobot:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
