"""
Mobile robot implementations.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..physics.base import BasePhysicsEngine
from ..robots.base import BaseRobot

logger = logging.getLogger(__name__)


class DifferentialDriveRobot(BaseRobot):
    """
    Differential drive mobile robot implementation.
    
    A two-wheeled mobile robot with independent wheel control.
    Common in robotics education and research.
    """
    
    def __init__(
        self,
        initial_pose: npt.NDArray[np.float32],
        physics_engine: Optional[BasePhysicsEngine] = None,
        wheel_radius: float = 0.1,
        wheel_base: float = 0.3,
        max_wheel_speed: float = 10.0,
    ) -> None:
        super().__init__(initial_pose)
        
        self.physics_engine = physics_engine
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_wheel_speed = max_wheel_speed
        
        # Robot parameters
        self.mass = 10.0  # kg
        self.inertia = 0.5  # kg*m^2
        
        # State variables
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        
        # Initialize in physics engine
        self._initialize_physics()
    
    def _initialize_physics(self) -> None:
        """Initialize robot in physics engine."""
        if self.physics_engine is not None:
            # Add ground plane
            self.physics_engine.add_ground_plane()
            
            # Create robot body (box shape)
            robot_size = np.array([0.4, 0.2, 0.1], dtype=np.float32)
            self.robot_id = self.physics_engine.add_box(
                position=self.current_pose[:3],
                orientation=self.current_pose[3:],
                size=robot_size,
                mass=self.mass,
            )
    
    def apply_action(self, action: npt.NDArray[np.float32]) -> None:
        """
        Apply wheel speed commands to the robot.
        
        Args:
            action: [left_wheel_speed, right_wheel_speed] in rad/s
        """
        if len(action) != 2:
            raise ValueError("Differential drive requires 2D action [left_speed, right_speed]")
        
        # Clamp wheel speeds
        self.left_wheel_speed = np.clip(action[0], -self.max_wheel_speed, self.max_wheel_speed)
        self.right_wheel_speed = np.clip(action[1], -self.max_wheel_speed, self.max_wheel_speed)
        
        # Update robot dynamics
        self._update_dynamics()
    
    def _update_dynamics(self) -> None:
        """Update robot dynamics based on wheel speeds."""
        # Convert wheel speeds to linear and angular velocities
        left_velocity = self.left_wheel_speed * self.wheel_radius
        right_velocity = self.right_wheel_speed * self.wheel_radius
        
        # Differential drive kinematics
        linear_velocity = (left_velocity + right_velocity) / 2.0
        angular_velocity = (right_velocity - left_velocity) / self.wheel_base
        
        # Update pose using simple integration
        dt = 0.01  # Assume 100Hz control loop
        
        # Update position
        x, y, z = self.current_pose[:3]
        roll, pitch, yaw = self.current_pose[3:]
        
        # Update x, y based on current heading
        x += linear_velocity * np.cos(yaw) * dt
        y += linear_velocity * np.sin(yaw) * dt
        
        # Update yaw
        yaw += angular_velocity * dt
        
        # Update pose
        self.current_pose = np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)
        
        # Update velocity
        self.current_velocity = np.array([
            linear_velocity * np.cos(yaw),
            linear_velocity * np.sin(yaw),
            0.0,  # No vertical velocity
            0.0,  # No roll velocity
            0.0,  # No pitch velocity
            angular_velocity,
        ], dtype=np.float32)
        
        # Update acceleration (simplified)
        self.current_acceleration = np.zeros(6, dtype=np.float32)
    
    def get_pose(self) -> npt.NDArray[np.float32]:
        """Get current robot pose."""
        return self.current_pose.copy()
    
    def get_velocity(self) -> npt.NDArray[np.float32]:
        """Get current robot velocity."""
        return self.current_velocity.copy()
    
    def get_acceleration(self) -> npt.NDArray[np.float32]:
        """Get current robot acceleration."""
        return self.current_acceleration.copy()
    
    def reset(self) -> None:
        """Reset robot to initial state."""
        self.current_pose = self.initial_pose.copy()
        self.current_velocity = np.zeros(6, dtype=np.float32)
        self.current_acceleration = np.zeros(6, dtype=np.float32)
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        
        # Reset in physics engine
        if self.physics_engine is not None and hasattr(self, 'robot_id'):
            self.physics_engine.set_object_pose(
                self.robot_id,
                self.current_pose[:3],
                self.current_pose[3:]
            )


class AckermannRobot(BaseRobot):
    """
    Ackermann steering mobile robot implementation.
    
    A car-like vehicle with front-wheel steering and rear-wheel drive.
    Common in autonomous vehicle research.
    """
    
    def __init__(
        self,
        initial_pose: npt.NDArray[np.float32],
        physics_engine: Optional[BasePhysicsEngine] = None,
        wheel_base: float = 2.5,
        track_width: float = 1.6,
        max_steering_angle: float = 0.5,
        max_speed: float = 10.0,
    ) -> None:
        super().__init__(initial_pose)
        
        self.physics_engine = physics_engine
        self.wheel_base = wheel_base
        self.track_width = track_width
        self.max_steering_angle = max_steering_angle
        self.max_speed = max_speed
        
        # Robot parameters
        self.mass = 1000.0  # kg (car-like)
        self.inertia = 100.0  # kg*m^2
        
        # State variables
        self.speed = 0.0
        self.steering_angle = 0.0
        
        # Initialize in physics engine
        self._initialize_physics()
    
    def _initialize_physics(self) -> None:
        """Initialize robot in physics engine."""
        if self.physics_engine is not None:
            # Add ground plane
            self.physics_engine.add_ground_plane()
            
            # Create robot body (box shape)
            robot_size = np.array([4.0, 2.0, 1.0], dtype=np.float32)
            self.robot_id = self.physics_engine.add_box(
                position=self.current_pose[:3],
                orientation=self.current_pose[3:],
                size=robot_size,
                mass=self.mass,
            )
    
    def apply_action(self, action: npt.NDArray[np.float32]) -> None:
        """
        Apply speed and steering commands to the robot.
        
        Args:
            action: [speed, steering_angle] in m/s and rad
        """
        if len(action) != 2:
            raise ValueError("Ackermann requires 2D action [speed, steering_angle]")
        
        # Clamp inputs
        self.speed = np.clip(action[0], -self.max_speed, self.max_speed)
        self.steering_angle = np.clip(action[1], -self.max_steering_angle, self.max_steering_angle)
        
        # Update robot dynamics
        self._update_dynamics()
    
    def _update_dynamics(self) -> None:
        """Update robot dynamics based on speed and steering."""
        dt = 0.01  # Assume 100Hz control loop
        
        # Ackermann steering kinematics
        if abs(self.steering_angle) < 1e-6:
            # Straight line motion
            angular_velocity = 0.0
        else:
            # Curved motion
            turning_radius = self.wheel_base / np.tan(self.steering_angle)
            angular_velocity = self.speed / turning_radius
        
        # Update pose
        x, y, z = self.current_pose[:3]
        roll, pitch, yaw = self.current_pose[3:]
        
        # Update position
        x += self.speed * np.cos(yaw) * dt
        y += self.speed * np.sin(yaw) * dt
        
        # Update yaw
        yaw += angular_velocity * dt
        
        # Update pose
        self.current_pose = np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)
        
        # Update velocity
        self.current_velocity = np.array([
            self.speed * np.cos(yaw),
            self.speed * np.sin(yaw),
            0.0,  # No vertical velocity
            0.0,  # No roll velocity
            0.0,  # No pitch velocity
            angular_velocity,
        ], dtype=np.float32)
        
        # Update acceleration (simplified)
        self.current_acceleration = np.zeros(6, dtype=np.float32)
    
    def get_pose(self) -> npt.NDArray[np.float32]:
        """Get current robot pose."""
        return self.current_pose.copy()
    
    def get_velocity(self) -> npt.NDArray[np.float32]:
        """Get current robot velocity."""
        return self.current_velocity.copy()
    
    def get_acceleration(self) -> npt.NDArray[np.float32]:
        """Get current robot acceleration."""
        return self.current_acceleration.copy()
    
    def reset(self) -> None:
        """Reset robot to initial state."""
        self.current_pose = self.initial_pose.copy()
        self.current_velocity = np.zeros(6, dtype=np.float32)
        self.current_acceleration = np.zeros(6, dtype=np.float32)
        self.speed = 0.0
        self.steering_angle = 0.0
        
        # Reset in physics engine
        if self.physics_engine is not None and hasattr(self, 'robot_id'):
            self.physics_engine.set_object_pose(
                self.robot_id,
                self.current_pose[:3],
                self.current_pose[3:]
            )
