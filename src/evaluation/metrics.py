"""
Evaluation metrics and benchmarking framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate various performance metrics for robotics systems.
    
    Provides comprehensive evaluation metrics for control, navigation,
    and learning performance.
    """
    
    def __init__(self) -> None:
        self.metrics_history: List[Dict[str, float]] = []
    
    def calculate_control_metrics(
        self,
        poses: npt.NDArray[np.float32],
        targets: npt.NDArray[np.float32],
        actions: npt.NDArray[np.float32],
        times: npt.NDArray[np.float32],
    ) -> Dict[str, float]:
        """
        Calculate control performance metrics.
        
        Args:
            poses: Robot poses over time
            targets: Target poses over time
            actions: Control actions over time
            times: Time array
            
        Returns:
            Dictionary of control metrics
        """
        metrics = {}
        
        # Position tracking error
        position_errors = np.linalg.norm(poses[:, :2] - targets[:, :2], axis=1)
        metrics['rmse_position'] = np.sqrt(np.mean(position_errors**2))
        metrics['mae_position'] = np.mean(position_errors)
        metrics['max_position_error'] = np.max(position_errors)
        
        # Settling time (time to reach 5% of final error)
        final_error = position_errors[-1]
        threshold = 0.05 * final_error if final_error > 0 else 0.01
        settling_indices = np.where(position_errors <= threshold)[0]
        if len(settling_indices) > 0:
            metrics['settling_time'] = times[settling_indices[0]]
        else:
            metrics['settling_time'] = times[-1]
        
        # Overshoot
        if len(position_errors) > 1:
            max_error = np.max(position_errors)
            final_error = position_errors[-1]
            if final_error > 0:
                metrics['overshoot_percent'] = ((max_error - final_error) / final_error) * 100
            else:
                metrics['overshoot_percent'] = 0.0
        else:
            metrics['overshoot_percent'] = 0.0
        
        # Control effort
        control_effort = np.sum(np.abs(actions), axis=1)
        metrics['total_control_effort'] = np.sum(control_effort)
        metrics['mean_control_effort'] = np.mean(control_effort)
        metrics['max_control_effort'] = np.max(control_effort)
        
        # Smoothness (jerk)
        if len(actions) > 2:
            jerk = np.diff(actions, axis=0)
            metrics['mean_jerk'] = np.mean(np.linalg.norm(jerk, axis=1))
            metrics['max_jerk'] = np.max(np.linalg.norm(jerk, axis=1))
        else:
            metrics['mean_jerk'] = 0.0
            metrics['max_jerk'] = 0.0
        
        return metrics
    
    def calculate_navigation_metrics(
        self,
        poses: npt.NDArray[np.float32],
        target_pose: npt.NDArray[np.float32],
        obstacles: Optional[List[npt.NDArray[np.float32]]] = None,
    ) -> Dict[str, float]:
        """
        Calculate navigation performance metrics.
        
        Args:
            poses: Robot trajectory
            target_pose: Final target position
            obstacles: List of obstacle positions
            
        Returns:
            Dictionary of navigation metrics
        """
        metrics = {}
        
        # Path length
        path_length = 0.0
        for i in range(1, len(poses)):
            path_length += np.linalg.norm(poses[i, :2] - poses[i-1, :2])
        metrics['path_length'] = path_length
        
        # Optimal path length (straight line)
        optimal_length = np.linalg.norm(target_pose[:2] - poses[0, :2])
        metrics['optimal_path_length'] = optimal_length
        metrics['path_efficiency'] = optimal_length / path_length if path_length > 0 else 0.0
        
        # Final position error
        final_error = np.linalg.norm(poses[-1, :2] - target_pose[:2])
        metrics['final_position_error'] = final_error
        
        # Success rate (reached within 0.5m of target)
        success_threshold = 0.5
        metrics['success'] = 1.0 if final_error <= success_threshold else 0.0
        
        # Collision detection
        if obstacles is not None:
            collision_count = 0
            robot_radius = 0.2  # Assume robot radius
            
            for pose in poses:
                for obstacle in obstacles:
                    distance = np.linalg.norm(pose[:2] - obstacle[:2])
                    if distance < robot_radius:
                        collision_count += 1
                        break
            
            metrics['collision_count'] = collision_count
            metrics['collision_rate'] = collision_count / len(poses)
        else:
            metrics['collision_count'] = 0
            metrics['collision_rate'] = 0.0
        
        return metrics
    
    def calculate_learning_metrics(
        self,
        rewards: npt.NDArray[np.float32],
        episode_lengths: npt.NDArray[np.int32],
        success_rates: npt.NDArray[np.float32],
    ) -> Dict[str, float]:
        """
        Calculate learning performance metrics.
        
        Args:
            rewards: Episode rewards
            episode_lengths: Episode lengths
            success_rates: Episode success rates
            
        Returns:
            Dictionary of learning metrics
        """
        metrics = {}
        
        # Reward statistics
        metrics['mean_reward'] = np.mean(rewards)
        metrics['std_reward'] = np.std(rewards)
        metrics['max_reward'] = np.max(rewards)
        metrics['min_reward'] = np.min(rewards)
        
        # Episode length statistics
        metrics['mean_episode_length'] = np.mean(episode_lengths)
        metrics['std_episode_length'] = np.std(episode_lengths)
        
        # Success rate
        metrics['mean_success_rate'] = np.mean(success_rates)
        metrics['final_success_rate'] = success_rates[-1] if len(success_rates) > 0 else 0.0
        
        # Sample efficiency (episodes to reach 80% success rate)
        if len(success_rates) > 0:
            target_success_rate = 0.8
            efficient_episodes = np.where(success_rates >= target_success_rate)[0]
            if len(efficient_episodes) > 0:
                metrics['sample_efficiency'] = efficient_episodes[0]
            else:
                metrics['sample_efficiency'] = len(success_rates)
        else:
            metrics['sample_efficiency'] = 0
        
        return metrics
    
    def calculate_sensor_metrics(
        self,
        sensor_data: List[Dict[str, Any]],
        ground_truth: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        Calculate sensor performance metrics.
        
        Args:
            sensor_data: Sensor measurements
            ground_truth: Ground truth data (if available)
            
        Returns:
            Dictionary of sensor metrics
        """
        metrics = {}
        
        if ground_truth is None:
            # Basic sensor statistics
            for sensor_name in sensor_data[0].keys():
                if sensor_data[0][sensor_name] is not None:
                    data = [s[sensor_name] for s in sensor_data if s[sensor_name] is not None]
                    if len(data) > 0:
                        data_array = np.array(data)
                        metrics[f'{sensor_name}_mean'] = np.mean(data_array)
                        metrics[f'{sensor_name}_std'] = np.std(data_array)
                        metrics[f'{sensor_name}_max'] = np.max(data_array)
                        metrics[f'{sensor_name}_min'] = np.min(data_array)
        else:
            # Sensor accuracy metrics
            for sensor_name in sensor_data[0].keys():
                if sensor_name in ground_truth[0] and sensor_data[0][sensor_name] is not None:
                    sensor_values = [s[sensor_name] for s in sensor_data if s[sensor_name] is not None]
                    gt_values = [s[sensor_name] for s in ground_truth if s[sensor_name] is not None]
                    
                    if len(sensor_values) > 0 and len(gt_values) > 0:
                        min_len = min(len(sensor_values), len(gt_values))
                        sensor_array = np.array(sensor_values[:min_len])
                        gt_array = np.array(gt_values[:min_len])
                        
                        errors = np.abs(sensor_array - gt_array)
                        metrics[f'{sensor_name}_rmse'] = np.sqrt(np.mean(errors**2))
                        metrics[f'{sensor_name}_mae'] = np.mean(errors)
                        metrics[f'{sensor_name}_max_error'] = np.max(errors)
        
        return metrics
    
    def create_leaderboard(
        self,
        results: List[Dict[str, Any]],
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Create a performance leaderboard.
        
        Args:
            results: List of experiment results
            metric_weights: Weights for different metrics
            
        Returns:
            DataFrame with ranked results
        """
        if not results:
            return pd.DataFrame()
        
        # Default weights
        if metric_weights is None:
            metric_weights = {
                'rmse_position': -1.0,  # Lower is better
                'settling_time': -1.0,  # Lower is better
                'path_efficiency': 1.0,  # Higher is better
                'success': 1.0,  # Higher is better
                'mean_reward': 1.0,  # Higher is better
            }
        
        # Calculate composite scores
        for result in results:
            score = 0.0
            for metric, weight in metric_weights.items():
                if metric in result['metrics']:
                    score += weight * result['metrics'][metric]
            result['composite_score'] = score
        
        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Create DataFrame
        df_data = []
        for i, result in enumerate(results):
            row = {
                'rank': i + 1,
                'method': result.get('method', f'Method_{i+1}'),
                'composite_score': result['composite_score'],
            }
            row.update(result['metrics'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df
    
    def save_results(self, results: List[Dict[str, Any]], filepath: str) -> None:
        """Save evaluation results to file."""
        df = self.create_leaderboard(results)
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
    
    def generate_report(
        self,
        results: List[Dict[str, Any]],
        output_file: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: List of experiment results
            output_file: Optional file to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("# Digital Twin Robotics Evaluation Report\n")
        
        # Summary statistics
        if results:
            report.append("## Summary Statistics\n")
            report.append(f"- Number of experiments: {len(results)}")
            report.append(f"- Best performing method: {results[0].get('method', 'Unknown')}")
            report.append(f"- Best composite score: {results[0].get('composite_score', 0.0):.3f}\n")
        
        # Detailed results
        report.append("## Detailed Results\n")
        df = self.create_leaderboard(results)
        report.append(df.to_string(index=False))
        
        # Recommendations
        report.append("\n## Recommendations\n")
        if results:
            best_result = results[0]
            report.append(f"- **Best Method**: {best_result.get('method', 'Unknown')}")
            report.append(f"- **Key Strengths**: High performance in multiple metrics")
            report.append("- **Areas for Improvement**: Consider parameter tuning")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text
