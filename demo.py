#!/usr/bin/env python3
"""
Digital Twin for Robotics - Main Demo Script

This script demonstrates the modernized digital twin system with
comprehensive simulation, control, and evaluation capabilities.

Usage:
    python demo.py --config config/mobile_robot.yaml --duration 10.0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from digital_twin import DigitalTwinRobot
from evaluation.metrics import MetricsCalculator
from visualization.plotter import Plotter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Digital Twin Robotics Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        default=[5.0, 5.0],
        help="Target position [x, y]"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save simulation data to file"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting Digital Twin Robotics Demo")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Duration: {args.duration}s")
    logger.info(f"Target: {args.target}")
    logger.info(f"Seed: {args.seed}")
    
    # Create digital twin
    try:
        with DigitalTwinRobot(
            config=args.config,
            seed=args.seed,
        ) as digital_twin:
            
            # Set target position
            target_pose = np.array([args.target[0], args.target[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            logger.info("Running simulation...")
            
            # Run simulation
            digital_twin.run_simulation(
                duration=args.duration,
                target_pose=target_pose,
            )
            
            # Get simulation data
            pose_history = np.array(digital_twin.pose_history)
            velocity_history = np.array(digital_twin.velocity_history)
            action_history = np.array(digital_twin.action_history)
            time_history = np.array(digital_twin.time_history)
            
            logger.info("Simulation completed")
            logger.info(f"Final position: {pose_history[-1, :2]}")
            logger.info(f"Target position: {target_pose[:2]}")
            logger.info(f"Final error: {np.linalg.norm(pose_history[-1, :2] - target_pose[:2]):.3f}m")
            
            # Calculate metrics
            logger.info("Calculating performance metrics...")
            metrics_calc = MetricsCalculator()
            
            # Create target array for metrics
            target_array = np.tile(target_pose[:2], (len(pose_history), 1))
            
            control_metrics = metrics_calc.calculate_control_metrics(
                poses=pose_history,
                targets=target_array,
                actions=action_history,
                times=time_history,
            )
            
            navigation_metrics = metrics_calc.calculate_navigation_metrics(
                poses=pose_history,
                target_pose=target_pose,
            )
            
            # Print metrics
            logger.info("Control Metrics:")
            for metric, value in control_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            logger.info("Navigation Metrics:")
            for metric, value in navigation_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Create visualization
            logger.info("Creating visualizations...")
            plotter = Plotter(save_plots=True)
            
            # Plot trajectory
            plotter.plot_trajectory(
                poses=pose_history,
                target_pose=target_pose,
            )
            
            # Plot control performance
            plotter.plot_control_performance(
                times=time_history,
                poses=pose_history,
                targets=target_array,
                actions=action_history,
            )
            
            # Save data if requested
            if args.save_data:
                data_file = f"assets/simulation_data_seed_{args.seed}.npz"
                digital_twin.save_data(data_file)
                logger.info(f"Data saved to {data_file}")
            
            logger.info("Demo completed successfully!")
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
