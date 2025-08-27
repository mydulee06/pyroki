import trimesh
import yourdfpy
from functools import partial

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float
import jaxlie

import warp as wp
from warp.jax_experimental.custom_call import jax_kernel

from .._robot import Robot

def uint64_to_uint32_pair(value):
    # Extract the lower 32 bits
    low = jnp.uint32(value & 0xFFFFFFFF)
    # Extract the upper 32 bits
    high = jnp.uint32(value >> 32)
    # return high, low
    return jnp.stack([high, low], axis=-1).astype(jnp.uint32)

def get_mesh_query_point():
    @wp.kernel
    def mesh_query_point(
        points: wp.array(dtype=wp.vec3),
        mesh_id_32: wp.array(dtype=wp.vec2ui),
        max_distance: wp.array(dtype=wp.float32),
        # outputs
        distance: wp.array(dtype=wp.float32),
        closest_direction: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()  # get the thread index

        point = points[tid]
        mesh_id = (wp.uint64(mesh_id_32[0][0]) << wp.uint64(32)) | wp.uint64(mesh_id_32[0][1])

        dist = max_distance[0]
        collide_result = wp.mesh_query_point(mesh_id, point, max_distance[0])
        if collide_result.result:
            sign = collide_result.sign
            # sign (float32): A value < 0 if query point is inside the mesh, >=0 otherwise.

            closest_point = wp.mesh_eval_position(
                mesh_id, collide_result.face, collide_result.u, collide_result.v
            )
            delta = closest_point - point
            dis_length = wp.length(delta)
            dist = sign * dis_length # - if inside, + if outside

        # Write the resulting signed distance into the output array.
        distance[tid] = dist
        closest_direction[tid] = delta
    return mesh_query_point


def get_mid_sole_link_pose(left_sole_link_pose, right_sole_link_pose):
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(
            (left_sole_link_pose.rotation().log() + right_sole_link_pose.rotation().log()) / 2
        ),
        translation=(left_sole_link_pose.translation() + right_sole_link_pose.translation()) / 2,
    )


class CollisionDetector:
    def __init__(self, robot: Robot, urdf: yourdfpy.URDF, obj_mesh: trimesh.Trimesh, max_query_points=100000):
        self.robot = robot
        self.left_sole_idx = robot.links.names.index("left_sole_link")
        self.right_sole_idx = robot.links.names.index("right_sole_link")

        self.urdf = urdf
        self._init_query_points(max_query_points)

        self.obj_mesh = obj_mesh
        vertices = wp.array(obj_mesh.vertices, dtype=wp.vec3)
        faces = wp.array(obj_mesh.faces.flatten(), dtype=int)
        mesh_wp = wp.Mesh(
            points=vertices,
            indices=faces,
        )
        self.mesh_id = uint64_to_uint32_pair(mesh_wp.id)[None]
        self.mesh_query_point_jax = jax_kernel(get_mesh_query_point())


    def _init_query_points(self, max_query_points):
        coll_link_ids = []
        for i, link in enumerate(self.urdf.link_map.values()):
            if len(link.collisions) > 0:
                coll_link_ids.append(i)
        self.coll_link_ids = jnp.array(coll_link_ids)

        coll_scene = self.urdf.collision_scene
        coll_geoms = list(coll_scene.geometry.values())
        coll_area = jnp.array([cg.area for cg in coll_geoms])
        n_samples = (max_query_points * (coll_area / coll_area.sum())).round().astype(jnp.int32)

        all_pts = []
        for cg, N in zip(coll_geoms, n_samples):
            pts, faces = trimesh.sample.sample_surface_even(cg, N)
            all_pts.append(pts)

        self.query_points = all_pts


    @partial(jax.jit, static_argnums=(0,))
    def mesh_query_point(
        self,
        points: jax.Array,
        max_distance: float = 1000.0,
    ):
        max_distance = jnp.array([max_distance], dtype=jnp.float32)
        signed_distance, direction = self.mesh_query_point_jax(points, self.mesh_id, max_distance)
        return signed_distance, direction


    @partial(jax.jit, static_argnums=(0,))
    def check_collision(self, cfg: jax.Array, terminal_mid_sole_pose: jaxlie.SE3):
        fk = self.robot.forward_kinematics(cfg)
        coll_link_tf = fk[self.coll_link_ids]

        left_sole_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(fk[self.left_sole_idx,:4]),
            translation=fk[self.left_sole_idx,4:],
        )
        right_sole_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(fk[self.right_sole_idx,:4]),
            translation=fk[self.right_sole_idx,4:],
        )
        mid_sole_pose = get_mid_sole_link_pose(left_sole_pose, right_sole_pose)

        coll_link_se3 = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(coll_link_tf[:,:4]),
            translation=coll_link_tf[:,4:],
        )
        coll_link_se3 = terminal_mid_sole_pose @ mid_sole_pose.inverse() @ coll_link_se3

        all_pts_tf = []
        for pts, tf in zip(self.query_points, coll_link_se3.parameters()):
            pts_tf = jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3(tf[:4]),
                translation=tf[4:],
            ).apply(pts)
            all_pts_tf.append(pts_tf)
        all_pts_tf = jnp.concat(all_pts_tf)

        # trimesh.Scene([trimesh.PointCloud(all_pts_tf), self.obj_mesh]).show()

        signed_distance, _ = self.mesh_query_point(all_pts_tf)
        is_collide = (signed_distance < 0).any()
        return is_collide


if __name__ == "__main__":
    robot = yourdfpy.URDF.load("examples/eetrack/robots/g1_29dof_rev_1_0_ver4_camera_mount_v4.urdf")
    mesh = trimesh.load("examples/eetrack/weld_objects/meshes/Circular_Body_Plate_110_80.obj")

    col_det = CollisionDetector(robot, mesh)
    key = jax.random.PRNGKey(0)
    query_points = jax.random.uniform(key, (100,3))
    max_distance = jnp.array([1000.0], dtype=jnp.float32)
    signed_distance,_ = col_det.mesh_query_point(query_points, max_distance)

    assert jnp.allclose(signed_distance, (-1 * trimesh.proximity.signed_distance(mesh, query_points)))
