"""
Visualization and plotting utilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class Plotter:
    """
    Real-time plotting and visualization system.
    
    Provides both matplotlib and plotly-based visualization
    for robot trajectories, sensor data, and control actions.
    """
    
    def __init__(
        self,
        real_time: bool = True,
        save_plots: bool = False,
        plot_type: str = "matplotlib",
    ) -> None:
        self.real_time = real_time
        self.save_plots = save_plots
        self.plot_type = plot_type
        
        # Data storage
        self.pose_history: List[npt.NDArray[np.float32]] = []
        self.velocity_history: List[npt.NDArray[np.float32]] = []
        self.action_history: List[npt.NDArray[np.float32]] = []
        self.sensor_history: List[Dict[str, Any]] = []
        self.time_history: List[float] = []
        
        # Initialize plotting
        if self.plot_type == "matplotlib":
            self._init_matplotlib()
        elif self.plot_type == "plotly":
            self._init_plotly()
    
    def _init_matplotlib(self) -> None:
        """Initialize matplotlib plotting."""
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Digital Twin Robot Visualization")
        
        # Configure subplots
        self.axes[0, 0].set_title("Robot Trajectory")
        self.axes[0, 0].set_xlabel("X Position (m)")
        self.axes[0, 0].set_ylabel("Y Position (m)")
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title("Velocity")
        self.axes[0, 1].set_xlabel("Time (s)")
        self.axes[0, 1].set_ylabel("Velocity (m/s)")
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title("Control Actions")
        self.axes[1, 0].set_xlabel("Time (s)")
        self.axes[1, 0].set_ylabel("Action")
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title("IMU Data")
        self.axes[1, 1].set_xlabel("Time (s)")
        self.axes[1, 1].set_ylabel("Acceleration (m/s²)")
        self.axes[1, 1].grid(True)
    
    def _init_plotly(self) -> None:
        """Initialize plotly plotting."""
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Robot Trajectory", "Velocity", "Control Actions", "IMU Data"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
    
    def update(
        self,
        pose: npt.NDArray[np.float32],
        velocity: npt.NDArray[np.float32],
        action: Optional[npt.NDArray[np.float32]],
        sensor_data: Dict[str, Any],
        simulation_time: float,
    ) -> None:
        """
        Update visualization with new data.
        
        Args:
            pose: Current robot pose
            velocity: Current robot velocity
            action: Current control action
            sensor_data: Sensor measurements
            simulation_time: Current simulation time
        """
        # Store data
        self.pose_history.append(pose.copy())
        self.velocity_history.append(velocity.copy())
        if action is not None:
            self.action_history.append(action.copy())
        self.sensor_history.append(sensor_data.copy())
        self.time_history.append(simulation_time)
        
        # Update plots
        if self.real_time:
            self._update_plots()
    
    def _update_plots(self) -> None:
        """Update all plots with current data."""
        if len(self.pose_history) < 2:
            return
        
        if self.plot_type == "matplotlib":
            self._update_matplotlib()
        elif self.plot_type == "plotly":
            self._update_plotly()
    
    def _update_matplotlib(self) -> None:
        """Update matplotlib plots."""
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Convert to numpy arrays
        poses = np.array(self.pose_history)
        velocities = np.array(self.velocity_history)
        times = np.array(self.time_history)
        
        # Plot trajectory
        self.axes[0, 0].plot(poses[:, 0], poses[:, 1], 'b-', linewidth=2, label='Trajectory')
        self.axes[0, 0].scatter(poses[-1, 0], poses[-1, 1], color='red', s=100, label='Current Position')
        self.axes[0, 0].set_title("Robot Trajectory")
        self.axes[0, 0].set_xlabel("X Position (m)")
        self.axes[0, 0].set_ylabel("Y Position (m)")
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)
        
        # Plot velocity
        linear_vel = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        self.axes[0, 1].plot(times, linear_vel, 'g-', linewidth=2, label='Linear Velocity')
        self.axes[0, 1].plot(times, velocities[:, 5], 'r-', linewidth=2, label='Angular Velocity')
        self.axes[0, 1].set_title("Velocity")
        self.axes[0, 1].set_xlabel("Time (s)")
        self.axes[0, 1].set_ylabel("Velocity")
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True)
        
        # Plot control actions
        if len(self.action_history) > 0:
            actions = np.array(self.action_history)
            self.axes[1, 0].plot(times[:len(actions)], actions[:, 0], 'b-', linewidth=2, label='Left Wheel')
            self.axes[1, 0].plot(times[:len(actions)], actions[:, 1], 'r-', linewidth=2, label='Right Wheel')
            self.axes[1, 0].set_title("Control Actions")
            self.axes[1, 0].set_xlabel("Time (s)")
            self.axes[1, 0].set_ylabel("Wheel Speed (rad/s)")
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True)
        
        # Plot IMU data
        if len(self.sensor_history) > 0 and 'imu' in self.sensor_history[-1]:
            imu_data = [s['imu']['accelerometer'] for s in self.sensor_history if 'imu' in s]
            if len(imu_data) > 0:
                imu_array = np.array(imu_data)
                imu_times = times[:len(imu_data)]
                self.axes[1, 1].plot(imu_times, imu_array[:, 0], 'r-', linewidth=2, label='X')
                self.axes[1, 1].plot(imu_times, imu_array[:, 1], 'g-', linewidth=2, label='Y')
                self.axes[1, 1].plot(imu_times, imu_array[:, 2], 'b-', linewidth=2, label='Z')
                self.axes[1, 1].set_title("IMU Accelerometer")
                self.axes[1, 1].set_xlabel("Time (s)")
                self.axes[1, 1].set_ylabel("Acceleration (m/s²)")
                self.axes[1, 1].legend()
                self.axes[1, 1].grid(True)
        
        # Refresh display
        plt.tight_layout()
        plt.pause(0.01)
    
    def _update_plotly(self) -> None:
        """Update plotly plots."""
        # This would implement plotly-based real-time updates
        # For now, we'll use matplotlib
        pass
    
    def save_plot(self, filename: str) -> None:
        """Save current plot to file."""
        if self.plot_type == "matplotlib":
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filename}")
    
    def close(self) -> None:
        """Close plotting system."""
        if self.plot_type == "matplotlib":
            plt.close(self.fig)
        logger.info("Plotter closed")
    
    def plot_trajectory(
        self,
        poses: npt.NDArray[np.float32],
        target_pose: Optional[npt.NDArray[np.float32]] = None,
        obstacles: Optional[List[npt.NDArray[np.float32]]] = None,
    ) -> None:
        """
        Plot robot trajectory with optional target and obstacles.
        
        Args:
            poses: Array of robot poses
            target_pose: Target position
            obstacles: List of obstacle positions
        """
        plt.figure(figsize=(10, 8))
        
        # Plot trajectory
        plt.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=2, label='Robot Trajectory')
        
        # Plot start and end points
        plt.scatter(poses[0, 0], poses[0, 1], color='green', s=100, label='Start')
        plt.scatter(poses[-1, 0], poses[-1, 1], color='red', s=100, label='End')
        
        # Plot target
        if target_pose is not None:
            plt.scatter(target_pose[0], target_pose[1], color='orange', s=100, label='Target')
        
        # Plot obstacles
        if obstacles is not None:
            for obs in obstacles:
                plt.scatter(obs[0], obs[1], color='black', s=50, marker='s', label='Obstacle')
        
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Robot Trajectory")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        if self.save_plots:
            self.save_plot("trajectory.png")
        
        plt.show()
    
    def plot_control_performance(
        self,
        times: npt.NDArray[np.float32],
        poses: npt.NDArray[np.float32],
        targets: npt.NDArray[np.float32],
        actions: npt.NDArray[np.float32],
    ) -> None:
        """
        Plot control performance metrics.
        
        Args:
            times: Time array
            poses: Robot poses
            targets: Target poses
            actions: Control actions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Position tracking
        axes[0, 0].plot(times, poses[:, 0], 'b-', label='X Position')
        axes[0, 0].plot(times, targets[:, 0], 'r--', label='X Target')
        axes[0, 0].set_title("X Position Tracking")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Position (m)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(times, poses[:, 1], 'b-', label='Y Position')
        axes[0, 1].plot(times, targets[:, 1], 'r--', label='Y Target')
        axes[0, 1].set_title("Y Position Tracking")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Position (m)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Control actions
        axes[1, 0].plot(times, actions[:, 0], 'g-', label='Left Wheel')
        axes[1, 0].set_title("Left Wheel Speed")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Speed (rad/s)")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(times, actions[:, 1], 'g-', label='Right Wheel')
        axes[1, 1].set_title("Right Wheel Speed")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Speed (rad/s)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if self.save_plots:
            self.save_plot("control_performance.png")
        
        plt.show()
