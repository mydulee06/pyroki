import os
import sys
import time
import yaml
import yourdfpy
import numpy as np
from pathlib import Path
from typing import TypedDict
import threading

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

import pyroki as pk
from pyroki.collision._robot_collision_custom import RobotCollisionV2

import viser
from viser.extras import ViserUrdf

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from trajectory_msgs.msg import JointTrajectoryPoint

from pyroki_ros.action import TrajOpt

from loguru import logger

jaxls.utils.logger.add(sys.stdout, level="SUCCESS")


class TrackingWeights(TypedDict):
    position_tracking: float
    orientation_tracking: float
    smoothness: float
    joint_limits: float


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


# Vectorized SE3 conversion function
def se3_from_pose(pose):
    # pose: (7,) [x, y, z, x, y, z, w] (xyzw)
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_quaternion_xyzw(pose[3:]), pose[:3]
    )


def compute_collision_costs(robot, coll_capsules, robot_cfg, active_idx_i, active_idx_j, safety_margin, collision_weight, link_indices_for_collision):
    Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg=robot_cfg)
    Ts_link_world_wxyz_xyz = Ts_link_world_wxyz_xyz[jnp.array(link_indices_for_collision)]
    import jaxlie
    coll_world = coll_capsules.transform(jaxlie.SE3(Ts_link_world_wxyz_xyz))
    from pyroki.collision._collision import pairwise_collide
    dist_matrix = pairwise_collide(coll_world, coll_world)
    dists = dist_matrix[active_idx_i, active_idx_j]
    costs = jnp.maximum(0, safety_margin - dists) * collision_weight
    return costs, dists


@jax.jit
def collision_cost_jax(
    robot_cfg,
    robot,
    coll_capsules,
    active_idx_i,
    active_idx_j,
    safety_margin,
    collision_weight,
    link_indices_for_collision
):
    costs, _ = compute_collision_costs(robot, coll_capsules, robot_cfg, active_idx_i, active_idx_j, safety_margin, collision_weight, link_indices_for_collision)
    return jnp.array([jnp.sum(costs)])


def make_solve_eetrack_optimization_jitted(robot, robot_collision, weights, max_iterations, collision_pairs, safety_margin):
    active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
    coll_capsules = robot_collision.coll
    link_indices_for_collision = [robot.links.names.index(name) for name in robot_collision.link_names]

    @jax.jit
    def solve(target_poses):
        timesteps = target_poses.shape[0]
        var_joints = robot.joint_var_cls(jnp.arange(timesteps))

        @jaxls.Cost.create_factory
        def path_tracking_cost_t(
            var_values: jaxls.VarValues,
            var_robot_cfg_t: jaxls.Var[jnp.ndarray],
            target_pose_t: jnp.ndarray,
        ) -> jax.Array:
            robot_cfg = var_values[var_robot_cfg_t]
            T_world_root = jaxlie.SE3.identity()
            end_effector_link_idx = robot.links.names.index("end_effector")
            fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
            ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
            T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
            ee_pose = T_world_root @ T_root_ee
            target_se3 = se3_from_pose(target_pose_t)
            error = (target_se3.inverse() @ ee_pose).log()
            position_error = error[:3]
            orientation_error = error[3:]
            weighted_error = jnp.hstack([
                position_error * weights["position_tracking"],
                orientation_error * weights["orientation_tracking"],
            ])
            return weighted_error

        @jaxls.Cost.create_factory
        def smoothness_cost_t(
            var_values: jaxls.VarValues,
            var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
            var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
        ) -> jax.Array:
            curr_cfg = var_values[var_robot_cfg_curr]
            prev_cfg = var_values[var_robot_cfg_prev]
            return (curr_cfg - prev_cfg) * weights["smoothness"]

        @jaxls.Cost.create_factory
        def collision_cost_t(
            var_values: jaxls.VarValues,
            var_robot_cfg_t: jaxls.Var[jnp.ndarray],
        ) -> jax.Array:
            robot_cfg = var_values[var_robot_cfg_t]
            return collision_cost_jax(
                robot_cfg,
                robot,
                coll_capsules,
                active_idx_i,
                active_idx_j,
                safety_margin,
                weights["collision"],
                link_indices_for_collision
            )

        costs = []
        for t in range(timesteps):
            costs.append(path_tracking_cost_t(var_joints[t], target_poses[t]))
            costs.append(pk.costs.limit_cost(robot, var_joints[t], weights["joint_limits"]))
            costs.append(collision_cost_t(var_joints[t]))
        for t in range(timesteps - 1):
            costs.append(smoothness_cost_t(var_joints[t+1], var_joints[t]))
        termination_config = jaxls.TerminationConfig(
            max_iterations=max_iterations,
            early_termination=False,
        )
        solution = (
            jaxls.LeastSquaresProblem(costs, [var_joints])
            .analyze()
            .solve(
                termination = termination_config,
            )
        )
        solved_joints = jnp.stack([solution[var_joints[t]] for t in range(timesteps)])
        return solved_joints

    return solve


def analyze_trajectory_optimized(robot, joints, target_poses, config, collision_pairs=None, robot_collision=None, safety_margin=None, collision_weight=None):
    """Optimized vectorized version of analyze_trajectory"""
    # Pre-compute constants outside of timestep loop
    end_effector_link_idx = robot.links.names.index("end_effector")
    T_world_root = jaxlie.SE3.identity()
    
    if collision_pairs is not None and robot_collision is not None:
        link_indices_for_collision = jnp.array([robot.links.names.index(name) for name in robot_collision.link_names])
        active_idx_i, active_idx_j = convert_collision_pairs_to_indices(collision_pairs, robot_collision)
    
    # Vectorized computation for single timestep
    def analyze_single_timestep(robot_cfg, target_pose):
        # Forward kinematics
        fk_poses_arr = robot.forward_kinematics(cfg=robot_cfg)
        ee_pose_in_root_arr = fk_poses_arr[end_effector_link_idx]
        T_root_ee = jaxlie.SE3(ee_pose_in_root_arr)
        ee_pose = T_world_root @ T_root_ee
        
        # Target pose conversion
        target_se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(target_pose[3:]),
            target_pose[:3]
        )
        
        # Error computation
        error = (target_se3.inverse() @ ee_pose).log()
        position_error = jnp.linalg.norm(error[:3])
        orientation_error = jnp.linalg.norm(error[3:])
        
        # Collision cost
        if collision_pairs is not None and robot_collision is not None and safety_margin is not None and collision_weight is not None:
            costs, _ = compute_collision_costs(
                robot, robot_collision.coll, robot_cfg,
                active_idx_i, active_idx_j,
                safety_margin, collision_weight,
                link_indices_for_collision
            )
            collision_cost = jnp.sum(costs)
        else:
            collision_cost = 0.0

        within_joint_limits = (
            (robot.joints.lower_limits < robot_cfg) &
            (robot_cfg < robot.joints.upper_limits)
        ).all()
            
        return position_error, orientation_error, collision_cost, within_joint_limits

    # Vectorize over all timesteps
    analyze_timestep_vmap = jax.vmap(analyze_single_timestep, in_axes=(0, 0))
    position_errors, orientation_errors, collision_costs, within_joint_limits = analyze_timestep_vmap(joints, target_poses)
    
    return position_errors, orientation_errors, collision_costs, within_joint_limits


def pose_stamped_msg_list_to_jax(pose_stamped_msg_list):
    poses_jax = []
    for pose_stamped_msg in pose_stamped_msg_list:
        pose_jax = jnp.array([
            pose_stamped_msg.pose.position.x,
            pose_stamped_msg.pose.position.y,
            pose_stamped_msg.pose.position.z,
            pose_stamped_msg.pose.orientation.x,
            pose_stamped_msg.pose.orientation.y,
            pose_stamped_msg.pose.orientation.z,
            pose_stamped_msg.pose.orientation.w,
        ])
        poses_jax.append(pose_jax)

    poses_jax = jnp.stack(poses_jax)
    return poses_jax

def append_dummy_target_ee_traj(target_ee_traj, num_step):
    if len(target_ee_traj) > num_step:
        return target_ee_traj

    # dummy_ee_traj = jnp.zeros((num_step - len(target_ee_traj), 7)).at[:,-1].set(1)
    dummy_ee_traj = target_ee_traj[-1:].repeat(num_step - len(target_ee_traj), 0)

    return jnp.concat([target_ee_traj, dummy_ee_traj])


class TrajOptActionServer(Node):
    def __init__(self):
        super().__init__('trajopt_action_server')
        self.get_logger().info("trajopt_action_server starts!")

        self._load_config()
        self._load_robot()
        self._compile_trajopt_fn()

        self._init_visualizer()
        threading.Thread(target=self._vis_trajectory_callback, daemon=True).start()

        self._action_server = ActionServer(
            self,
            TrajOpt,
            'trajopt',
            self.execute_trajopt,
        )


    def _load_config(self):
        self.asset_dir = Path(__file__).parent / "eetrack"
        config_file = self.asset_dir / "config_ros.yaml"
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)


    def _load_robot(self):
        urdf_path = self.config['robot']['urdf_path']
        urdf_obj = yourdfpy.URDF.load(urdf_path)
        sit_terminal_states = np.load(self.config['robot']['sit_terminal_states_path'])
        idx = np.abs(sit_terminal_states["target_height"] - self.config['robot']['sit_target_height']).argmin()
        joint_pos = sit_terminal_states["joint_state"][idx, 0]
        lab2yourdf = [np.where(sit_terminal_states["lab_joint"] == jn)[0].item() for jn in urdf_obj.actuated_joint_names]
        urdf_obj.update_cfg(joint_pos[lab2yourdf])
        for joint in urdf_obj.robot.joints:
            if joint.name in urdf_obj.actuated_joint_names and joint.name not in self.config['robot']['movable_joints']:
                joint.type = "fixed"
                joint.origin = urdf_obj.get_transform(joint.child, joint.parent)
        self.modified_urdf = yourdfpy.URDF(urdf_obj.robot, mesh_dir=Path(urdf_path).parent)

        # Robot
        self.robot = pk.Robot.from_urdf(self.modified_urdf)

        # Robot collision
        collision_cfg = self.config.get('collision', {})
        ignore_pairs = tuple(tuple(pair) for pair in collision_cfg.get('ignore_pairs', []))
        exclude_links = tuple(collision_cfg.get('exclude_links', []))
        self.robot_collision = RobotCollisionV2.from_urdf(
            self.modified_urdf,
            user_ignore_pairs=ignore_pairs,
            ignore_immediate_adjacents=collision_cfg.get('ignore_adjacent_links', True),
            exclude_links=exclude_links
        )


    def _compile_trajopt_fn(self):
        weights = TrackingWeights(
            position_tracking=self.config['weights']['position_tracking'],
            orientation_tracking=self.config['weights']['orientation_tracking'],
            smoothness=self.config['weights']['smoothness'],
            joint_limits=self.config['weights']['joint_limits'],
            collision=self.config['weights'].get('collision', 1.0),
        )
        max_iterations = self.config.get('optimization', {}).get('max_iterations', 30)
        safety_margin = self.config['collision'].get('safety_margin', 0.05)
        collision_pairs = self.config.get('collision_pairs', [])

        self.solve_fn = make_solve_eetrack_optimization_jitted(
            self.robot,
            self.robot_collision,
            weights,
            max_iterations,
            collision_pairs,
            safety_margin,
        )

        @jax.jit
        def analyze_fn_jit(joints, target_poses):
            return analyze_trajectory_optimized(
                self.robot,
                joints,
                target_poses,
                self.config,
                collision_pairs,
                self.robot_collision,
                safety_margin,
                weights['collision']
            )
        self.validation_fn = analyze_fn_jit

        # Warmpup functions
        self.max_traj_len = self.config["welding_path"]["num_timesteps"]
        dummy_ee_traj = jnp.zeros((self.max_traj_len, 7)).at[:,-1].set(1)
        dumm_joint_traj = self.solve_fn(dummy_ee_traj)

        self.validation_fn(dumm_joint_traj, dummy_ee_traj)


    def _check_success(self, joint_traj, target_ee_traj, max_step):
        position_errors, orientation_errors, collision_costs, within_joint_limits = \
            self.validation_fn(joint_traj, target_ee_traj)

        pos_tol = self.config['tolerance']['position_error']
        ori_tol = self.config['tolerance']['orientation_error']
        collision_threshold = 0.001

        max_position_error = position_errors[:max_step].max().item()
        max_orientation_error = orientation_errors[:max_step].max().item()
        max_collision_cost = collision_costs[:max_step].max().item()
        all_within_joint_limits = within_joint_limits.all().item()

        self.get_logger().info(f"Max position error: {max_position_error:.4f}")
        self.get_logger().info(f"Max orientation error: {max_orientation_error:.4f}")
        self.get_logger().info(f"Max collision cost: {max_collision_cost:.4f}")
        self.get_logger().info(f"All within joint limit: {all_within_joint_limits}")
        
        # Vectorized failure detection
        position_failed = max_position_error > pos_tol
        orientation_failed = max_orientation_error > ori_tol
        collision_failed = max_collision_cost > collision_threshold
        joint_limit_failed = not all_within_joint_limits
        success = not (position_failed or orientation_failed or collision_failed or joint_limit_failed)

        return success


    def execute_trajopt(self, goal_handle):
        self.get_logger().info('Executing goal...')

        target_ee_traj = goal_handle.request.ee_traj

        target_ee_traj_jax = pose_stamped_msg_list_to_jax(target_ee_traj.poses)

        valid_max_step = self.max_traj_len
        if len(target_ee_traj_jax) < self.max_traj_len:
            # Append dummy ee traj for avoiding re-jit.
            valid_max_step = len(target_ee_traj_jax)
            target_ee_traj_jax = append_dummy_target_ee_traj(target_ee_traj_jax, self.max_traj_len)
        if len(target_ee_traj_jax) > self.max_traj_len:
            self.get_logger().warning(f'Recieved EE trajectory is larger than maximum length {self.max_traj_len}. Truncate EE traj to {self.max_traj_len}.')
            target_ee_traj_jax = target_ee_traj_jax[:self.max_traj_len]

        joint_traj_jax = self.solve_fn(target_ee_traj_jax)

        success = self._check_success(joint_traj_jax, target_ee_traj_jax, valid_max_step)

        stamp = self.get_clock().now().to_msg()
        frame_id = target_ee_traj.header.frame_id

        result = TrajOpt.Result()
        if success:
            result.success = True
            result.joint_traj.header.stamp = stamp
            result.joint_traj.header.frame_id = frame_id

            result.joint_traj.joint_names = self.robot.joints.actuated_names

            joint_traj = joint_traj_jax[:valid_max_step].tolist()
            for joint in joint_traj:
                point = JointTrajectoryPoint(positions=joint)
                result.joint_traj.points.append(point)
            goal_handle.succeed()
            self.get_logger().info(f'Succeed to find successful joint trajectories!!')
        else:
            result.success = False
            result.joint_traj.header.stamp = stamp
            result.joint_traj.header.frame_id = frame_id
            result.error_message = "TODO"
            goal_handle.abort()
            self.get_logger().info(f'Fail to find successful joint trajectories!!')

        self._set_vis_trajectory(target_ee_traj_jax, joint_traj_jax)

        return result


    def _init_visualizer(self):
        self.server = viser.ViserServer()

        # GUI
        self.playing = self.server.gui.add_checkbox("playing", True)
        self.timestep_slider = self.server.gui.add_slider("timestep", 0, self.max_traj_len - 1, 1, 0)

        # Scene
        self.server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis = ViserUrdf(self.server, self.modified_urdf, root_node_name="/base")

        self.target_pose_vis = self.server.scene.add_frame(
            "/target_pose",
            axes_length=0.1,
            axes_radius=0.002,
        )

        self.vis = False


    def _set_vis_trajectory(self, target_ee_traj, joint_traj):
        self.vis = True

        self.target_ee_traj_vis = target_ee_traj
        self.joint_traj_vis = joint_traj


    def _vis_trajectory_callback(self):
        while True:
            time.sleep(0.02)

            if not self.vis:
                continue

            with self.server.atomic():
                if self.playing.value:
                    self.timestep_slider.value = (self.timestep_slider.value + 1) % self.max_traj_len
                tstep = self.timestep_slider.value

                robot_cfg = self.joint_traj_vis[tstep]
                self.urdf_vis.update_cfg(np.array(robot_cfg))

                target_ee = self.target_ee_traj_vis[tstep]
                self.target_pose_vis.position = np.array(target_ee[:3])
                self.target_pose_vis.wxyz = np.roll(np.array(target_ee[3:]), 1)


def main(args=None):
    rclpy.init(args=args)
    trajopt_server = TrajOptActionServer()
    try:
        rclpy.spin(trajopt_server)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()