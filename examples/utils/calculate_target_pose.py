#!/usr/bin/env python3
"""
Target Mid-Sole Pose Calculator

This module contains functions to calculate target mid-sole poses in world frame
given welding object poses and search space constraints.

Key Functions:
- compose_transforms: Compose two SE(3) transforms
- calculate_target_mid_sole_pose: Calculate deterministic mid-sole pose from object pose
- sample_mid_sole_pose_from_search_space: Sample random mid-sole pose within search space
- sample_mid_sole_poses_batch: Batch sampling of mid-sole poses
"""

import numpy as np
import jax.numpy as jnp
import jaxlie
from typing import Tuple, Dict, Any, List
import warnings


def compose_transforms(x1: float, y1: float, z1: float, yaw1: float,
                      x2: float, y2: float, z2: float, yaw2: float) -> Tuple[float, float, float, float]:
    """
    Compose two SE(3) transforms: T_result = T1 @ T2
    
    Args:
        x1, y1, z1, yaw1: First transform (translation + yaw rotation)
        x2, y2, z2, yaw2: Second transform (translation + yaw rotation)
        
    Returns:
        Tuple of (x, y, z, yaw) representing the composed transform
    """
    so3_1 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw1)
    T1 = jaxlie.SE3.from_rotation_and_translation(so3_1, jnp.array([x1, y1, z1]))
    so3_2 = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw2)
    T2 = jaxlie.SE3.from_rotation_and_translation(so3_2, jnp.array([x2, y2, z2]))
    T_result = T1 @ T2
    translation = T_result.translation()
    rpy = T_result.rotation().as_rpy_radians()
    return float(translation[0]), float(translation[1]), float(translation[2]), float(rpy[2])


def calculate_target_mid_sole_pose(object_x: float, object_y: float, object_z: float, object_yaw: float,
                                  relative_x: float, relative_y: float, relative_z: float, relative_yaw: float) -> Tuple[float, float, float, float]:
    """
    Calculate target mid-sole pose given welding object pose and relative offset.
    
    Args:
        object_x, object_y, object_z, object_yaw: Welding object pose in world frame
        relative_x, relative_y, relative_z, relative_yaw: Relative offset from object to mid-sole
        
    Returns:
        Tuple of (mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw) in world frame
    """
    return compose_transforms(
        object_x, object_y, object_z, object_yaw,
        relative_x, relative_y, relative_z, relative_yaw
    )


def sample_mid_sole_pose_from_search_space(object_x: float, object_y: float, object_z: float, object_yaw: float,
                                          search_space: Dict[str, Any], 
                                          verbose: bool = False) -> Tuple[float, float, float, float]:
    """
    Sample a random mid-sole pose from search space relative to welding object.
    
    Args:
        object_x, object_y, object_z, object_yaw: Welding object pose in world frame
        search_space: Dictionary containing sampling ranges
            - 'x_range': [min_x, max_x] for relative x offset
            - 'y_range': [min_y, max_y] for relative y offset  
            - 'z_height': relative z offset (default: 0.0)
            - 'angle_range': [min_yaw, max_yaw] for relative yaw offset
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw) in world frame
    """
    if verbose:
        print(f"🔄 Sampling mid-sole pose from search space...")
        print(f"📍 Target welding object (world): x={object_x:.3f}, y={object_y:.3f}, z={object_z:.3f}, yaw={object_yaw:.3f}")
    
    # Sample relative pose from search space
    relative_x = np.random.uniform(*search_space['x_range'])
    relative_y = np.random.uniform(*search_space['y_range'])
    relative_z = search_space.get('z_height', 0.0)
    relative_yaw = np.random.uniform(*search_space['angle_range'])
    
    if verbose:
        print(f"   Random relative pose: x={relative_x:.3f}, y={relative_y:.3f}, z={relative_z:.3f}, yaw={relative_yaw:.3f}")
    
    # Calculate target mid-sole pose
    mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = calculate_target_mid_sole_pose(
        object_x, object_y, object_z, object_yaw,
        relative_x, relative_y, relative_z, relative_yaw
    )
    
    if verbose:
        print(f"   -> Mid-sole pose (world): x={mid_sole_x:.3f}, y={mid_sole_y:.3f}, z={mid_sole_z:.3f}, yaw={mid_sole_yaw:.3f}")
    
    return mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw


def sample_mid_sole_poses_batch(object_x: float, object_y: float, object_z: float, object_yaw: float,
                               search_space: Dict[str, Any], 
                               batch_size: int) -> np.ndarray:
    """
    Sample multiple mid-sole poses in batch from search space.
    
    Args:
        object_x, object_y, object_z, object_yaw: Welding object pose in world frame
        search_space: Dictionary containing sampling ranges
        batch_size: Number of samples to generate
        
    Returns:
        numpy array of shape (batch_size, 4) containing [x, y, z, yaw] for each sample
    """
    # Sample relative poses from search space
    relative_x = np.random.uniform(*search_space['x_range'], size=batch_size)
    relative_y = np.random.uniform(*search_space['y_range'], size=batch_size)
    relative_z = np.full(batch_size, search_space.get('z_height', 0.0))
    relative_yaw = np.random.uniform(*search_space['angle_range'], size=batch_size)
    
    # Calculate target mid-sole poses for all samples
    mid_sole_poses = []
    for i in range(batch_size):
        mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = calculate_target_mid_sole_pose(
            object_x, object_y, object_z, object_yaw,
            relative_x[i], relative_y[i], relative_z[i], relative_yaw[i]
        )
        mid_sole_poses.append([mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw])
    
    return np.array(mid_sole_poses)


def validate_search_space(search_space: Dict[str, Any]) -> bool:
    """
    Validate search space dictionary format.
    
    Args:
        search_space: Dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['x_range', 'y_range', 'angle_range']
    
    for key in required_keys:
        if key not in search_space:
            warnings.warn(f"Missing required key '{key}' in search_space")
            return False
        
        if not isinstance(search_space[key], (list, tuple)) or len(search_space[key]) != 2:
            warnings.warn(f"'{key}' should be a list/tuple of 2 values [min, max]")
            return False
        
        if search_space[key][0] >= search_space[key][1]:
            warnings.warn(f"'{key}' min value should be less than max value")
            return False
    
    # Check optional z_height
    if 'z_height' in search_space:
        if not isinstance(search_space['z_height'], (int, float)):
            warnings.warn("'z_height' should be a numeric value")
            return False
    
    return True


def get_search_space_bounds(search_space: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """
    Extract bounds from search space for easier access.
    
    Args:
        search_space: Search space dictionary
        
    Returns:
        Dictionary containing bounds for each dimension
    """
    if not validate_search_space(search_space):
        raise ValueError("Invalid search space format")
    
    return {
        'x': tuple(search_space['x_range']),
        'y': tuple(search_space['y_range']),
        'z': (search_space.get('z_height', 0.0), search_space.get('z_height', 0.0)),
        'yaw': tuple(search_space['angle_range'])
    }


def create_grid_samples(object_x: float, object_y: float, object_z: float, object_yaw: float,
                       search_space: Dict[str, Any], 
                       grid_size: int = 10) -> np.ndarray:
    """
    Create grid samples of mid-sole poses within search space.
    
    Args:
        object_x, object_y, object_z, object_yaw: Welding object pose in world frame
        search_space: Dictionary containing sampling ranges
        grid_size: Number of samples per dimension (total samples = grid_size^3)
        
    Returns:
        numpy array of shape (grid_size^3, 4) containing [x, y, z, yaw] for each sample
    """
    bounds = get_search_space_bounds(search_space)
    
    # Create grid for relative coordinates
    rel_x_grid = np.linspace(bounds['x'][0], bounds['x'][1], grid_size)
    rel_y_grid = np.linspace(bounds['y'][0], bounds['y'][1], grid_size)
    rel_yaw_grid = np.linspace(bounds['yaw'][0], bounds['yaw'][1], grid_size)
    rel_z = search_space.get('z_height', 0.0)
    
    # Generate all combinations
    rel_x_mesh, rel_y_mesh, rel_yaw_mesh = np.meshgrid(rel_x_grid, rel_y_grid, rel_yaw_grid, indexing='ij')
    
    # Flatten and calculate target poses
    rel_x_flat = rel_x_mesh.flatten()
    rel_y_flat = rel_y_mesh.flatten()
    rel_yaw_flat = rel_yaw_mesh.flatten()
    
    mid_sole_poses = []
    for i in range(len(rel_x_flat)):
        mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw = calculate_target_mid_sole_pose(
            object_x, object_y, object_z, object_yaw,
            rel_x_flat[i], rel_y_flat[i], rel_z, rel_yaw_flat[i]
        )
        mid_sole_poses.append([mid_sole_x, mid_sole_y, mid_sole_z, mid_sole_yaw])
    
    return np.array(mid_sole_poses)


# Example usage and test functions
def test_compose_transforms():
    """Test compose_transforms function with simple cases"""
    print("Testing compose_transforms...")
    
    # Test identity composition
    x, y, z, yaw = compose_transforms(1.0, 2.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0)
    assert np.isclose([x, y, z, yaw], [1.0, 2.0, 0.3, 0.5]).all(), "Identity composition failed"
    
    # Test translation composition
    x, y, z, yaw = compose_transforms(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert np.isclose([x, y, z, yaw], [2.0, 0.0, 0.0, 0.0]).all(), "Translation composition failed"
    
    print("✅ compose_transforms tests passed")


def test_sample_functions():
    """Test sampling functions"""
    print("Testing sampling functions...")
    
    search_space = {
        'x_range': [-0.5, 0.5],
        'y_range': [-0.3, 0.3],
        'z_height': 0.0,
        'angle_range': [-0.2, 0.2]
    }
    
    # Test single sample
    x, y, z, yaw = sample_mid_sole_pose_from_search_space(1.0, 2.0, 0.3, 0.1, search_space)
    print(f"Single sample: ({x:.3f}, {y:.3f}, {z:.3f}, {yaw:.3f})")
    
    # Test batch sampling
    batch_samples = sample_mid_sole_poses_batch(1.0, 2.0, 0.3, 0.1, search_space, 5)
    print(f"Batch samples shape: {batch_samples.shape}")
    print(f"First batch sample: {batch_samples[0]}")
    
    # Test grid sampling
    grid_samples = create_grid_samples(1.0, 2.0, 0.3, 0.1, search_space, 3)
    print(f"Grid samples shape: {grid_samples.shape}")
    
    print("✅ Sampling function tests passed")


if __name__ == "__main__":
    test_compose_transforms()
    test_sample_functions()
    print("🎉 All tests passed!")
