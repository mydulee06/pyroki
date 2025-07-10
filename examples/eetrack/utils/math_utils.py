import jax
import jax.numpy as jnp
from typing import Literal

@jax.jit
def _sqrt_positive_part(x: jnp.ndarray) -> jnp.ndarray:
    """
    Returns jnp.sqrt(jnp.maximum(0, x)) but with a zero sub-gradient where x is 0.
    JAX implementation of the PyTorch function.
    """
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)


@jax.jit
def quat_from_matrix(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Convert rotations given as rotation matrices to quaternions.
    JAX implementation of the PyTorch function.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    # Unpack matrix elements
    m = matrix.reshape(batch_dim + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [m[..., i] for i in range(9)]

    q_abs = _sqrt_positive_part(
        jnp.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    # We produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = jnp.stack(
        [
            jnp.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            jnp.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20], axis=-1),
            jnp.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21], axis=-1),
            jnp.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2], axis=-1),
        ],
        axis=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = jnp.array(0.1, dtype=q_abs.dtype)
    quat_candidates = quat_by_rijk / (2.0 * jnp.maximum(flr, q_abs[..., None]))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    best_indices = jnp.argmax(q_abs, axis=-1)
    one_hot_mask = jax.nn.one_hot(best_indices, num_classes=4, dtype=quat_candidates.dtype)
    selected_quats = jnp.sum(quat_candidates * one_hot_mask[..., None], axis=-2)

    return selected_quats

@jax.jit
def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the conjugate of a quaternion. JAX implementation.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion. Shape is (..., 4).
    """
    # Conjugate is (w, -x, -y, -z)
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])

@jax.jit
def quat_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Multiply two quaternions together. JAX implementation.
    This version supports broadcasting.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions. Shape is broadcasted (..., 4).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Using the same efficient multiplication formula
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return jnp.stack([w, x, y, z], axis=-1)

@jax.jit
def quat_unique(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a unit quaternion to a standard form where the real part is non-negative.
    JAX implementation.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Standardized quaternions. Shape is (..., 4).
    """
    return jnp.where(q[..., 0:1] < 0, -q, q)

@jax.jit
def axis_angle_from_quat(quat: jnp.ndarray, eps: float = 1.0e-6) -> jnp.ndarray:
    """
    Convert rotations given as quaternions to axis/angle. JAX implementation.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
    """
    quat = quat_unique(quat) # Ensure w is non-negative
    mag = jnp.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = jnp.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle

    # Taylor expansion for sin(x)/x when x is small
    small_angle_cond = jnp.abs(angle) <= eps
    sin_half_angles_over_angles = jnp.where(
        small_angle_cond,
        0.5 - angle**2 / 48.0,
        jnp.sin(half_angle) / angle
    )

    return quat[..., 1:4] / sin_half_angles_over_angles[..., None]

@jax.jit
def safe_rotvec2quat(rotvec: jnp.ndarray, form: Literal["xyzw", "wxyz"] = "wxyz") -> jnp.ndarray:
    """
    Convert a rotation vector to a quaternion. JAX implementation.
    Handles the singularity at angle = 0.
    """
    angle = jnp.linalg.norm(rotvec, axis=-1)

    # Taylor expansion for sin(angle/2)/angle for small angles
    small_angle_cond = angle <= 1e-3
    # Avoid division by zero for the large angle case
    angle_safe = jnp.where(angle == 0, 1.0, angle)

    scale = jnp.where(
        small_angle_cond,
        0.5 - angle**2 / 48.0 + angle**4 / 3840.0,
        jnp.sin(angle_safe / 2.0) / angle_safe
    )

    xyz = scale[..., None] * rotvec
    w = jnp.cos(angle / 2.0)[..., None]

    if form == "wxyz":
        return jnp.concatenate([w, xyz], axis=-1)
    else: # "xyzw"
        return jnp.concatenate([xyz, w], axis=-1)

@jax.jit
def slerp(q0: jnp.ndarray, q1: jnp.ndarray, steps: jnp.ndarray) -> jnp.ndarray:
    """
    Spherical linear interpolation between two quaternions. JAX implementation.

    Args:
        q0: The starting quaternion. Shape is (B, 4).
        q1: The ending quaternion. Shape is (B, 4).
        steps: The interpolation steps. Shape is (S,).

    Returns:
        The interpolated quaternions. Shape is (B, S, 4).
    """
    # The relative rotation between q0 and q1
    quat_diff = quat_mul(quat_conjugate(q0), q1)
    quat_diff = quat_unique(quat_diff)

    # Convert to axis-angle representation (the logarithm)
    rot_vec = axis_angle_from_quat(quat_diff)  # Shape: (B, 3)

    # Scale the rotation by the steps for each element in the batch
    diff = rot_vec * steps[..., None]

    # Convert back to quaternion
    diff_quat = safe_rotvec2quat(diff)

    # Apply the interpolated rotation to the starting quaternion
    # q0 -> (B, 1, 4) to broadcast with diff_quat
    return quat_mul(q0, diff_quat)


@jax.jit
def lerp(start: jax.Array, end: jax.Array, weight: jax.Array):
    return start + weight * (end - start)

