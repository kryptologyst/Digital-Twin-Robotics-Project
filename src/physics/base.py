"""
Base classes for digital twin components.

This module provides abstract base classes for robots, controllers,
sensors, and physics engines that form the core of the digital twin system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt


class BaseRobot(ABC):
    """Abstract base class for robot implementations."""
    
    def __init__(self, initial_pose: npt.NDArray[np.float32]) -> None:
        self.initial_pose = initial_pose.copy()
        self.current_pose = initial_pose.copy()
        self.current_velocity = np.zeros(6, dtype=np.float32)
        self.current_acceleration = np.zeros(6, dtype=np.float32)
    
    @abstractmethod
    def apply_action(self, action: npt.NDArray[np.float32]) -> None:
        """Apply control action to the robot."""
        pass
    
    @abstractmethod
    def get_pose(self) -> npt.NDArray[np.float32]:
        """Get current robot pose."""
        pass
    
    @abstractmethod
    def get_velocity(self) -> npt.NDArray[np.float32]:
        """Get current robot velocity."""
        pass
    
    @abstractmethod
    def get_acceleration(self) -> npt.NDArray[np.float32]:
        """Get current robot acceleration."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset robot to initial state."""
        pass


class BaseController(ABC):
    """Abstract base class for control systems."""
    
    def __init__(self) -> None:
        self.reset()
    
    @abstractmethod
    def compute_action(
        self,
        current_pose: npt.NDArray[np.float32],
        current_velocity: npt.NDArray[np.float32],
        target_pose: Optional[npt.NDArray[np.float32]] = None,
        target_velocity: Optional[npt.NDArray[np.float32]] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> npt.NDArray[np.float32]:
        """Compute control action."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset controller state."""
        pass


class BaseSensor(ABC):
    """Abstract base class for sensors."""
    
    def __init__(self, name: str) -> None:
        self.name = name
    
    @abstractmethod
    def get_measurement(
        self,
        robot_pose: npt.NDArray[np.float32],
        robot_velocity: npt.NDArray[np.float32],
        simulation_time: float,
    ) -> Any:
        """Get sensor measurement."""
        pass


class BasePhysicsEngine(ABC):
    """Abstract base class for physics simulation engines."""
    
    def __init__(self, timestep: float) -> None:
        self.timestep = timestep
    
    @abstractmethod
    def step(self) -> None:
        """Execute one physics simulation step."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset physics simulation."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close physics engine and clean up resources."""
        pass
