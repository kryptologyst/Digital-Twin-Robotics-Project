"""
IMU sensor implementation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from ..sensors.base import BaseSensor
from ..utils.device import get_device

logger = logging.getLogger(__name__)


class IMUSensor(BaseSensor):
    """
    Inertial Measurement Unit (IMU) sensor.
    
    Provides accelerometer, gyroscope, and magnetometer measurements
    with realistic noise models.
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        bias_std: float = 0.001,
        device: Optional[str] = None,
        name: str = "imu",
    ) -> None:
        super().__init__(name)
        
        self.device = get_device(device)
        self.noise_std = noise_std
        self.bias_std = bias_std
        
        # Sensor biases (constant offset)
        self.accel_bias = np.random.normal(0, bias_std, 3)
        self.gyro_bias = np.random.normal(0, bias_std, 3)
        self.mag_bias = np.random.normal(0, bias_std, 3)
        
        # Earth's magnetic field (simplified)
        self.mag_field = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        logger.info(f"IMU sensor initialized with noise_std={noise_std}")
    
    def get_measurement(
        self,
        robot_pose: npt.NDArray[np.float32],
        robot_velocity: npt.NDArray[np.float32],
        simulation_time: float,
    ) -> dict[str, npt.NDArray[np.float32]]:
        """
        Get IMU measurement.
        
        Args:
            robot_pose: Robot pose [x, y, z, roll, pitch, yaw]
            robot_velocity: Robot velocity [vx, vy, vz, wx, wy, wz]
            simulation_time: Current simulation time
            
        Returns:
            Dictionary containing accelerometer, gyroscope, and magnetometer data
        """
        # Extract orientation
        roll, pitch, yaw = robot_pose[3:6]
        
        # Extract angular velocity
        wx, wy, wz = robot_velocity[3:6]
        
        # Accelerometer measurement
        # Gravity vector in world frame
        gravity_world = np.array([0, 0, -9.81], dtype=np.float32)
        
        # Rotate gravity to body frame
        gravity_body = self._rotate_vector(gravity_world, roll, pitch, yaw)
        
        # Add noise and bias
        accel_noise = np.random.normal(0, self.noise_std, 3)
        accelerometer = gravity_body + self.accel_bias + accel_noise
        
        # Gyroscope measurement
        gyro_noise = np.random.normal(0, self.noise_std, 3)
        gyroscope = np.array([wx, wy, wz], dtype=np.float32) + self.gyro_bias + gyro_noise
        
        # Magnetometer measurement
        # Rotate magnetic field to body frame
        mag_body = self._rotate_vector(self.mag_field, roll, pitch, yaw)
        
        # Add noise and bias
        mag_noise = np.random.normal(0, self.noise_std, 3)
        magnetometer = mag_body + self.mag_bias + mag_noise
        
        return {
            "accelerometer": accelerometer.astype(np.float32),
            "gyroscope": gyroscope.astype(np.float32),
            "magnetometer": magnetometer.astype(np.float32),
            "timestamp": np.float32(simulation_time),
        }
    
    def _rotate_vector(
        self,
        vector: npt.NDArray[np.float32],
        roll: float,
        pitch: float,
        yaw: float,
    ) -> npt.NDArray[np.float32]:
        """Rotate vector from world frame to body frame."""
        # Create rotation matrices
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        # Roll rotation matrix
        R_x = np.array([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r],
        ], dtype=np.float32)
        
        # Pitch rotation matrix
        R_y = np.array([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p],
        ], dtype=np.float32)
        
        # Yaw rotation matrix
        R_z = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        
        # Combined rotation matrix (ZYX order)
        R = R_z @ R_y @ R_x
        
        # Rotate vector
        rotated_vector = R @ vector
        
        return rotated_vector.astype(np.float32)
    
    def reset(self) -> None:
        """Reset sensor biases."""
        self.accel_bias = np.random.normal(0, self.bias_std, 3)
        self.gyro_bias = np.random.normal(0, self.bias_std, 3)
        self.mag_bias = np.random.normal(0, self.bias_std, 3)
        logger.debug("IMU sensor reset")
