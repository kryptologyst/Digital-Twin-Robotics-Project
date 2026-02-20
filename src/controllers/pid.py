"""
PID controller implementation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from ..controllers.base import BaseController
from ..utils.device import get_device

logger = logging.getLogger(__name__)


class PIDController(BaseController):
    """
    Proportional-Integral-Derivative controller.
    
    A classical control method widely used in robotics for position
    and velocity control tasks.
    """
    
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.05,
        device: Optional[str] = None,
        output_limits: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        
        self.device = get_device(device)
        
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Output limits
        self.output_limits = output_limits
        
        # Controller state
        self.previous_error = None
        self.integral = None
        
        # Initialize state
        self.reset()
    
    def compute_action(
        self,
        current_pose: npt.NDArray[np.float32],
        current_velocity: npt.NDArray[np.float32],
        target_pose: Optional[npt.NDArray[np.float32]] = None,
        target_velocity: Optional[npt.NDArray[np.float32]] = None,
        sensor_data: Optional[dict] = None,
    ) -> npt.NDArray[np.float32]:
        """
        Compute PID control action.
        
        Args:
            current_pose: Current robot pose [x, y, z, roll, pitch, yaw]
            current_velocity: Current robot velocity
            target_pose: Desired robot pose
            target_velocity: Desired robot velocity
            sensor_data: Additional sensor data (unused for PID)
            
        Returns:
            Control action
        """
        if target_pose is None:
            # No target, return zero action
            return np.zeros(2, dtype=np.float32)
        
        # Compute position error
        position_error = target_pose[:2] - current_pose[:2]  # Only x, y for mobile robots
        
        # Compute velocity error
        if target_velocity is not None:
            velocity_error = target_velocity[:2] - current_velocity[:2]
        else:
            velocity_error = np.zeros(2, dtype=np.float32)
        
        # Total error (position + velocity)
        error = position_error + velocity_error
        
        # Initialize previous error if first call
        if self.previous_error is None:
            self.previous_error = np.zeros_like(error)
        
        # Initialize integral if first call
        if self.integral is None:
            self.integral = np.zeros_like(error)
        
        # PID computation
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative_error = error - self.previous_error
        d_term = self.kd * derivative_error
        
        # Total PID output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update previous error
        self.previous_error = error.copy()
        
        # Convert to robot-specific action format
        # For differential drive: [left_wheel_speed, right_wheel_speed]
        # For Ackermann: [speed, steering_angle]
        
        # Simple conversion: assume differential drive for now
        # Linear velocity = (left + right) / 2
        # Angular velocity = (right - left) / wheel_base
        
        linear_velocity = output[0]  # Use x-error for linear velocity
        angular_velocity = output[1]  # Use y-error for angular velocity
        
        # Convert to wheel speeds (assuming wheel_base = 0.3)
        wheel_base = 0.3
        left_wheel_speed = (linear_velocity - angular_velocity * wheel_base / 2) / 0.1  # wheel_radius = 0.1
        right_wheel_speed = (linear_velocity + angular_velocity * wheel_base / 2) / 0.1
        
        action = np.array([left_wheel_speed, right_wheel_speed], dtype=np.float32)
        
        return action
    
    def reset(self) -> None:
        """Reset controller state."""
        self.previous_error = None
        self.integral = None
        logger.debug("PID controller reset")
    
    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """Update PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        logger.info(f"PID gains updated: kp={kp}, ki={ki}, kd={kd}")
    
    def get_gains(self) -> tuple[float, float, float]:
        """Get current PID gains."""
        return self.kp, self.ki, self.kd
