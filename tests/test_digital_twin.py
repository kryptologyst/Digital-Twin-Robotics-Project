"""
Test suite for digital twin robotics.

Provides comprehensive unit and integration tests for all components.
"""

import pytest
import numpy as np
import numpy.typing as npt

from src.digital_twin import DigitalTwinRobot
from src.controllers.pid import PIDController
from src.sensors.imu import IMUSensor
from src.robots.mobile import DifferentialDriveRobot
from src.physics.pybullet_engine import PyBulletEngine
from src.evaluation.metrics import MetricsCalculator


class TestDigitalTwinRobot:
    """Test cases for DigitalTwinRobot class."""
    
    def test_initialization(self):
        """Test digital twin initialization."""
        digital_twin = DigitalTwinRobot(
            robot_type="mobile",
            platform="differential_drive",
            seed=42
        )
        
        assert digital_twin.robot_type == "mobile"
        assert digital_twin.platform == "differential_drive"
        assert digital_twin.simulation_time == 0.0
        assert digital_twin.is_running == False
    
    def test_reset(self):
        """Test digital twin reset functionality."""
        digital_twin = DigitalTwinRobot(seed=42)
        
        # Run a few steps
        for _ in range(10):
            digital_twin.update_digital_twin()
        
        # Reset
        digital_twin.reset()
        
        assert digital_twin.simulation_time == 0.0
        assert len(digital_twin.pose_history) == 0
        assert len(digital_twin.velocity_history) == 0


class TestPIDController:
    """Test cases for PIDController class."""
    
    def test_initialization(self):
        """Test PID controller initialization."""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05)
        
        assert controller.kp == 1.0
        assert controller.ki == 0.1
        assert controller.kd == 0.05
    
    def test_compute_action(self):
        """Test PID control action computation."""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05)
        
        current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        current_velocity = np.zeros(6, dtype=np.float32)
        target_pose = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        action = controller.compute_action(
            current_pose=current_pose,
            current_velocity=current_velocity,
            target_pose=target_pose,
        )
        
        assert action.shape == (2,)
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.float32
    
    def test_reset(self):
        """Test PID controller reset."""
        controller = PIDController()
        
        # Compute action to set internal state
        current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        current_velocity = np.zeros(6, dtype=np.float32)
        target_pose = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        controller.compute_action(current_pose, current_velocity, target_pose)
        
        # Reset
        controller.reset()
        
        assert controller.previous_error is None
        assert controller.integral is None


class TestIMUSensor:
    """Test cases for IMUSensor class."""
    
    def test_initialization(self):
        """Test IMU sensor initialization."""
        sensor = IMUSensor(noise_std=0.01, name="test_imu")
        
        assert sensor.name == "test_imu"
        assert sensor.noise_std == 0.01
        assert len(sensor.accel_bias) == 3
        assert len(sensor.gyro_bias) == 3
        assert len(sensor.mag_bias) == 3
    
    def test_get_measurement(self):
        """Test IMU measurement generation."""
        sensor = IMUSensor(noise_std=0.01, name="test_imu")
        
        robot_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        robot_velocity = np.zeros(6, dtype=np.float32)
        simulation_time = 0.0
        
        measurement = sensor.get_measurement(
            robot_pose=robot_pose,
            robot_velocity=robot_velocity,
            simulation_time=simulation_time,
        )
        
        assert "accelerometer" in measurement
        assert "gyroscope" in measurement
        assert "magnetometer" in measurement
        assert "timestamp" in measurement
        
        assert len(measurement["accelerometer"]) == 3
        assert len(measurement["gyroscope"]) == 3
        assert len(measurement["magnetometer"]) == 3


class TestDifferentialDriveRobot:
    """Test cases for DifferentialDriveRobot class."""
    
    def test_initialization(self):
        """Test differential drive robot initialization."""
        initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        robot = DifferentialDriveRobot(initial_pose)
        
        assert robot.wheel_radius == 0.1
        assert robot.wheel_base == 0.3
        assert robot.max_wheel_speed == 10.0
    
    def test_apply_action(self):
        """Test applying control action to robot."""
        initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        robot = DifferentialDriveRobot(initial_pose)
        
        action = np.array([1.0, 1.0], dtype=np.float32)
        robot.apply_action(action)
        
        assert robot.left_wheel_speed == 1.0
        assert robot.right_wheel_speed == 1.0
    
    def test_get_pose(self):
        """Test getting robot pose."""
        initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        robot = DifferentialDriveRobot(initial_pose)
        
        pose = robot.get_pose()
        
        assert pose.shape == (6,)
        assert isinstance(pose, np.ndarray)
        assert pose.dtype == np.float32


class TestMetricsCalculator:
    """Test cases for MetricsCalculator class."""
    
    def test_initialization(self):
        """Test metrics calculator initialization."""
        calc = MetricsCalculator()
        
        assert len(calc.metrics_history) == 0
    
    def test_calculate_control_metrics(self):
        """Test control metrics calculation."""
        calc = MetricsCalculator()
        
        # Create test data
        poses = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        targets = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        actions = np.array([[1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        times = np.array([0.0, 1.0], dtype=np.float32)
        
        metrics = calc.calculate_control_metrics(poses, targets, actions, times)
        
        assert "rmse_position" in metrics
        assert "mae_position" in metrics
        assert "settling_time" in metrics
        assert "total_control_effort" in metrics
    
    def test_calculate_navigation_metrics(self):
        """Test navigation metrics calculation."""
        calc = MetricsCalculator()
        
        # Create test data
        poses = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [2.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target_pose = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        metrics = calc.calculate_navigation_metrics(poses, target_pose)
        
        assert "path_length" in metrics
        assert "optimal_path_length" in metrics
        assert "path_efficiency" in metrics
        assert "final_position_error" in metrics
        assert "success" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
