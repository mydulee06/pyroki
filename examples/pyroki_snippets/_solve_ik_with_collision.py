"""
Solves the basic IK problem with collision avoidance.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def convert_collision_pairs_to_indices(collision_pairs, robot_collision):
    link_names = robot_collision.link_names
    link_name_to_idx = {name: i for i, name in enumerate(link_names)}
    active_idx_i = []
    active_idx_j = []
    for pair in collision_pairs:
        if pair[0] in link_name_to_idx and pair[1] in link_name_to_idx:
            active_idx_i.append(link_name_to_idx[pair[0]])
            active_idx_j.append(link_name_to_idx[pair[1]])
    return jnp.array(active_idx_i), jnp.array(active_idx_j)


def solve_ik_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    collision_pairs,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: Sequence[str]. Length: num_targets.
        position: ArrayLike. Shape: (num_targets, 3), or (3,).
        wxyz: ArrayLike. Shape: (num_targets, 4), or (4,).

    Returns:
        cfg: ArrayLike. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_idx = robot.links.names.index(target_link_name)

    T_world_targets = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, coll)
    link_indices_for_collision = [robot.links.names.index(name) for name in coll.link_names]
    cfg = _solve_ik_with_collision_jax(
        robot,
        coll,
        world_coll_list,
        T_world_targets,
        jnp.array(target_link_idx),
        active_idx_i,
        active_idx_j,
        link_indices_for_collision,
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_with_collision_jax(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
    active_idx_i,
    active_idx_j,
    link_indices_for_collision,
) -> jax.Array:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var = robot.joint_var_cls(0)
    vars = [joint_var]

    @jaxls.Cost.create_factory
    def collision_cost_t(
        var_values: jaxls.VarValues,
        var_robot_cfg_t: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg_t]
        return pk.costs.collision_cost_jax(
            robot_cfg,
            robot,
            coll.coll,
            active_idx_i,
            active_idx_j,
            safety_margin=0.001,
            collision_weight=10.0,
            link_indices_for_collision=link_indices_for_collision,
        )

    # Weights and margins defined directly in factors
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var=joint_var,
            weight=100.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll=coll,
            joint_var=joint_var,
            margin=0.02,
            weight=5.0,
        ),
    ]
    costs.extend(
        [
            pk.costs.world_collision_cost(
                robot, coll, joint_var, world_coll, 0.05, 10.0
            )
            for world_coll in world_coll_list
        ]
    )

    sol = (
        jaxls.LeastSquaresProblem(costs, vars)
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky")
    )
    return sol[joint_var]
