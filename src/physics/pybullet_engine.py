"""
PyBullet physics engine implementation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pybullet as p

from .base import BasePhysicsEngine

logger = logging.getLogger(__name__)


class PyBulletEngine(BasePhysicsEngine):
    """
    PyBullet physics simulation engine.
    
    Provides realistic physics simulation with collision detection,
    dynamics, and multi-body systems.
    """
    
    def __init__(
        self,
        timestep: float = 0.01,
        gravity: Optional[List[float]] = None,
        real_time: bool = False,
        gui: bool = False,
    ) -> None:
        super().__init__(timestep)
        
        self.gravity = gravity or [0, 0, -9.81]
        self.real_time = real_time
        self.gui = gui
        
        # PyBullet connection
        self.client_id: Optional[int] = None
        
        # Initialize physics
        self._initialize_physics()
    
    def _initialize_physics(self) -> None:
        """Initialize PyBullet physics simulation."""
        try:
            # Connect to PyBullet
            if self.gui:
                self.client_id = p.connect(p.GUI)
            else:
                self.client_id = p.connect(p.DIRECT)
            
            # Set physics parameters
            p.setGravity(*self.gravity, physicsClientId=self.client_id)
            p.setTimeStep(self.timestep, physicsClientId=self.client_id)
            
            # Set real-time simulation
            if self.real_time:
                p.setRealTimeSimulation(1, physicsClientId=self.client_id)
            
            logger.info("PyBullet physics engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PyBullet: {e}")
            raise
    
    def step(self) -> None:
        """Execute one physics simulation step."""
        if self.client_id is not None:
            p.stepSimulation(physicsClientId=self.client_id)
    
    def reset(self) -> None:
        """Reset physics simulation."""
        if self.client_id is not None:
            p.resetSimulation(physicsClientId=self.client_id)
            p.setGravity(*self.gravity, physicsClientId=self.client_id)
            p.setTimeStep(self.timestep, physicsClientId=self.client_id)
    
    def close(self) -> None:
        """Close physics engine and clean up resources."""
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = None
            logger.info("PyBullet physics engine closed")
    
    def add_ground_plane(self) -> int:
        """Add a ground plane to the simulation."""
        if self.client_id is None:
            raise RuntimeError("Physics engine not initialized")
        
        ground_id = p.loadURDF(
            "plane.urdf",
            physicsClientId=self.client_id
        )
        return ground_id
    
    def add_box(
        self,
        position: npt.NDArray[np.float32],
        orientation: npt.NDArray[np.float32],
        size: npt.NDArray[np.float32],
        mass: float = 1.0,
    ) -> int:
        """Add a box to the simulation."""
        if self.client_id is None:
            raise RuntimeError("Physics engine not initialized")
        
        # Create box shape
        box_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=size / 2,
            physicsClientId=self.client_id
        )
        
        # Create multi-body
        box_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=box_shape,
            basePosition=position.tolist(),
            baseOrientation=orientation.tolist(),
            physicsClientId=self.client_id
        )
        
        return box_id
    
    def add_sphere(
        self,
        position: npt.NDArray[np.float32],
        radius: float,
        mass: float = 1.0,
    ) -> int:
        """Add a sphere to the simulation."""
        if self.client_id is None:
            raise RuntimeError("Physics engine not initialized")
        
        # Create sphere shape
        sphere_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self.client_id
        )
        
        # Create multi-body
        sphere_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=sphere_shape,
            basePosition=position.tolist(),
            physicsClientId=self.client_id
        )
        
        return sphere_id
    
    def get_object_pose(self, object_id: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Get object position and orientation."""
        if self.client_id is None:
            raise RuntimeError("Physics engine not initialized")
        
        pos, orn = p.getBasePositionAndOrientation(
            object_id,
            physicsClientId=self.client_id
        )
        
        return np.array(pos, dtype=np.float32), np.array(orn, dtype=np.float32)
    
    def set_object_pose(
        self,
        object_id: int,
        position: npt.NDArray[np.float32],
        orientation: npt.NDArray[np.float32],
    ) -> None:
        """Set object position and orientation."""
        if self.client_id is None:
            raise RuntimeError("Physics engine not initialized")
        
        p.resetBasePositionAndOrientation(
            object_id,
            position.tolist(),
            orientation.tolist(),
            physicsClientId=self.client_id
        )
    
    def apply_force(
        self,
        object_id: int,
        force: npt.NDArray[np.float32],
        position: Optional[npt.NDArray[np.float32]] = None,
    ) -> None:
        """Apply force to an object."""
        if self.client_id is None:
            raise RuntimeError("Physics engine not initialized")
        
        if position is None:
            p.applyExternalForce(
                object_id,
                -1,  # Apply to base
                force.tolist(),
                [0, 0, 0],  # No position offset
                p.WORLD_FRAME,
                physicsClientId=self.client_id
            )
        else:
            p.applyExternalForce(
                object_id,
                -1,  # Apply to base
                force.tolist(),
                position.tolist(),
                p.WORLD_FRAME,
                physicsClientId=self.client_id
            )
    
    def get_object_velocity(self, object_id: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Get object linear and angular velocity."""
        if self.client_id is None:
            raise RuntimeError("Physics engine not initialized")
        
        linear_vel, angular_vel = p.getBaseVelocity(
            object_id,
            physicsClientId=self.client_id
        )
        
        return (
            np.array(linear_vel, dtype=np.float32),
            np.array(angular_vel, dtype=np.float32)
        )
