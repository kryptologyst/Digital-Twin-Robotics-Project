"""
Robots package for digital twin robotics.

Provides robot platform implementations including mobile robots, manipulators, and aerial vehicles.
"""

from .base import BaseRobot
from .mobile import DifferentialDriveRobot, AckermannRobot

__all__ = ["BaseRobot", "DifferentialDriveRobot", "AckermannRobot"]
