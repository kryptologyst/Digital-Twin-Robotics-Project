"""
Project 677: Digital Twin for Robotics - MODERNIZED IMPLEMENTATION

This file contains the original simple implementation alongside the new modernized system.
The original code has been refactored into a comprehensive, production-ready digital twin
framework with proper architecture, type hints, and extensive functionality.

ORIGINAL IMPLEMENTATION (Simple):
- Basic robot simulation with position tracking
- Simple sensor noise model
- Basic visualization with matplotlib
- Limited to 2D mobile robot simulation

MODERNIZED IMPLEMENTATION (Comprehensive):
- Multi-platform robot support (mobile, manipulator, aerial)
- Physics-based simulation with PyBullet/MuJoCo
- Comprehensive sensor suite (IMU, camera, LiDAR)
- Advanced control systems (PID, LQR, MPC, RL)
- Real-time visualization and evaluation metrics
- ROS 2 integration and production-ready architecture
- Extensive testing and safety features

To run the modernized system, use:
    python demo.py --config config/mobile_robot.yaml --duration 10.0

For the original simple implementation, see the code below.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modernized system to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import modernized components
from digital_twin import DigitalTwinRobot
from controllers.pid import PIDController
from sensors.imu import IMUSensor
from evaluation.metrics import MetricsCalculator

# =============================================================================
# ORIGINAL SIMPLE IMPLEMENTATION (Preserved for Reference)
# =============================================================================

class SimpleDigitalTwinRobot:
    """
    Original simple digital twin implementation.
    
    This is the original implementation that has been preserved for reference.
    The modernized system provides much more comprehensive functionality.
    """
    
    def __init__(self, initial_position=(0, 0), target_position=(5, 5), map_size=(10, 10)):
        self.position = np.array(initial_position)  # Robot's initial position
        self.target_position = np.array(target_position)  # Goal position
        self.velocity = np.array([0.1, 0.1])  # Robot's initial velocity
        self.map_size = map_size  # Size of the map (grid)
        self.sensor_noise = 0.1  # Simulated sensor noise for measurements
        self.history = []  # Store the history of positions for visualization
 
    def move(self):
        """
        Simulate robot movement based on its velocity.
        """
        self.position += self.velocity
        # Ensure the robot stays within the bounds of the map
        self.position = np.clip(self.position, [0, 0], np.array(self.map_size) - 1)
 
    def get_sensor_data(self):
        """
        Simulate sensor data (distance from robot to target with added noise).
        :return: Simulated sensor data with noise (distance to target)
        """
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        noisy_measurement = distance_to_target + np.random.randn() * self.sensor_noise
        return noisy_measurement
 
    def update(self):
        """
        Update the robot's state (position and sensor data).
        """
        self.move()  # Move the robot
        sensor_data = self.get_sensor_data()  # Get sensor data (distance to target)
        self.history.append(self.position.copy())  # Record the position
        return sensor_data
 
    def plot(self):
        """
        Visualize the robot's trajectory and the digital twin behavior.
        """
        history_array = np.array(self.history)
        plt.figure(figsize=(8, 8))
        plt.plot(history_array[:, 0], history_array[:, 1], label="Robot Path", color='blue')
        plt.scatter(self.target_position[0], self.target_position[1], color='red', s=100, label="Target Position")
        plt.scatter(self.position[0], self.position[1], color='green', s=100, label="Current Position")
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])
        plt.title("Digital Twin Robot Simulation (Original)")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()

# =============================================================================
# MODERNIZED IMPLEMENTATION DEMO
# =============================================================================

def run_original_demo():
    """Run the original simple implementation."""
    print("Running ORIGINAL simple digital twin implementation...")
    
    # Initialize the digital twin robot and simulate its movement
    digital_twin = SimpleDigitalTwinRobot(initial_position=(0, 0), target_position=(8, 8), map_size=(10, 10))
    
    # Simulate the robot's movement and sensor updates over 50 steps
    for step in range(50):
        sensor_data = digital_twin.update()  # Update the robot's position and get sensor data
        if step % 10 == 0:  # Plot every 10 steps
            digital_twin.plot()  # Plot the current position and trajectory

def run_modernized_demo():
    """Run the modernized comprehensive implementation."""
    print("Running MODERNIZED digital twin implementation...")
    print("This demonstrates the new comprehensive system with:")
    print("- Physics-based simulation")
    print("- Advanced sensors (IMU)")
    print("- PID control system")
    print("- Performance evaluation")
    print("- Real-time visualization")
    
    try:
        # Create modernized digital twin
        digital_twin = DigitalTwinRobot(
            robot_type="mobile",
            platform="differential_drive",
            config="config/mobile_robot.yaml",
            seed=42
        )
        
        # Set target position
        target_pose = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        print(f"Target position: {target_pose[:2]}")
        
        # Run simulation for 5 seconds
        digital_twin.run_simulation(
            duration=5.0,
            target_pose=target_pose,
        )
        
        # Get results
        pose_history = np.array(digital_twin.pose_history)
        final_error = np.linalg.norm(pose_history[-1, :2] - target_pose[:2])
        
        print(f"Final position: {pose_history[-1, :2]}")
        print(f"Final error: {final_error:.3f}m")
        
        # Calculate performance metrics
        metrics_calc = MetricsCalculator()
        target_array = np.tile(target_pose[:2], (len(pose_history), 1))
        
        control_metrics = metrics_calc.calculate_control_metrics(
            poses=pose_history,
            targets=target_array,
            actions=np.array(digital_twin.action_history),
            times=np.array(digital_twin.time_history),
        )
        
        print("\nPerformance Metrics:")
        print(f"  RMSE Position: {control_metrics['rmse_position']:.4f}m")
        print(f"  Settling Time: {control_metrics['settling_time']:.2f}s")
        print(f"  Control Effort: {control_metrics['total_control_effort']:.2f}")
        
        # Clean up
        digital_twin.close()
        
    except Exception as e:
        print(f"Modernized demo failed: {e}")
        print("This is expected if dependencies are not installed.")
        print("Run 'pip install -e .' to install the modernized system.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DIGITAL TWIN FOR ROBOTICS - COMPARISON DEMO")
    print("=" * 60)
    
    # Run original implementation
    print("\n1. ORIGINAL IMPLEMENTATION:")
    run_original_demo()
    
    print("\n" + "=" * 60)
    
    # Run modernized implementation
    print("\n2. MODERNIZED IMPLEMENTATION:")
    run_modernized_demo()
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print("\nThe modernized system provides:")
    print("✓ Physics-based simulation with PyBullet")
    print("✓ Comprehensive sensor models (IMU, camera, LiDAR)")
    print("✓ Advanced control systems (PID, LQR, MPC)")
    print("✓ Real-time visualization and evaluation")
    print("✓ ROS 2 integration and production architecture")
    print("✓ Extensive testing and safety features")
    print("✓ Multi-platform robot support")
    print("\nFor full functionality, install dependencies and run:")
    print("  python demo.py --config config/mobile_robot.yaml")


