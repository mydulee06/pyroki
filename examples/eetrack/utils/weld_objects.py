import trimesh
from pathlib import Path

import jax
import jaxlie
import jax.numpy as jnp
from .math_utils import lerp, quat_from_matrix, slerp


class WeldObject:
    def __init__(
            self,
            object_dir: str,
            object_name: str,
            segment_length_per_timestep: float = 0.0002,
            segment_normal_deg_range: tuple[float, float] = (-1.0, 1.0),
            offset_length: float = 0.02,
            approach_deg: float = 45.0,
        ):
        self.object_dir = Path(object_dir)
        self.mesh_path = self.object_dir / "meshes" / (object_name + ".obj")
        self.mesh = trimesh.load(self.mesh_path, force="mesh")

        self.edge_data_path = self.object_dir / "edges" / (object_name + ".npz")
        self.init_edge_data(self.edge_data_path)
        self.init_welding_path(offset_length, approach_deg, segment_length_per_timestep, segment_normal_deg_range)


    @property
    def trimesh(self):
        return self.mesh

    def init_edge_data(self, edge_data_path):
        edge_data = jnp.load(edge_data_path)

        self.edge_vertices = edge_data["vertices"]
        self.edge_normals = edge_data["normals"]
        self.edge_z_dir = jnp.zeros_like(self.edge_normals).at[:,2].set(1.0)

    def init_welding_path(self, offset_length, approach_deg, segment_length_per_timestep, segment_normal_deg_range):
        approach_rad = approach_deg/180.0*jnp.pi
        offset_vec = (
            offset_length*jnp.cos(approach_rad)*self.edge_normals +
            offset_length*jnp.sin(approach_rad)*self.edge_z_dir
        )
        line_vertices = self.edge_vertices + offset_vec

        line_vertices_length = jnp.linalg.norm(
            jnp.diff(
                line_vertices,
                axis=0,
                append=line_vertices[:1],
            ),
            axis=1,
        )
        num_segments = (line_vertices_length / segment_length_per_timestep).round().astype(jnp.int32)
        lerp_weight = jnp.concat([jnp.linspace(0, 1, num=n+1)[1:] for n in num_segments])

        pos_lerp_start = jnp.repeat(line_vertices, num_segments, axis=0)
        pos_lerp_end = jnp.repeat(jnp.concat([line_vertices[1:], line_vertices[:1]]), num_segments, axis=0)
        eetrack_poss_b = lerp(pos_lerp_start, pos_lerp_end, lerp_weight[:,None])

        eetrack_x_dirs = -offset_vec
        eetrack_x_dirs /= jnp.linalg.norm(eetrack_x_dirs, axis=1, keepdims=True)
        eetrack_y_dirs = jnp.stack([-eetrack_x_dirs[:,1], eetrack_x_dirs[:,0], jnp.zeros_like(eetrack_x_dirs[:,1])], axis=1)
        eetrack_y_dirs /= jnp.linalg.norm(eetrack_y_dirs, axis=1, keepdims=True)
        eetrack_z_dirs = jnp.cross(eetrack_x_dirs, eetrack_y_dirs)
        eetrack_mats = jnp.stack([eetrack_x_dirs, eetrack_y_dirs, eetrack_z_dirs], axis=2)
        eetrack_quats = quat_from_matrix(eetrack_mats)

        # Assume closed loop
        quat_lerp_start = jnp.repeat(eetrack_quats, num_segments, axis=0)
        quat_lerp_end = jnp.repeat(jnp.concat([eetrack_quats[1:], eetrack_quats[:1]]), num_segments, axis=0)
        eetrack_quats_b = slerp(quat_lerp_start, quat_lerp_end, lerp_weight)

        normal_yaw = jnp.arctan2(self.edge_normals[:,1], self.edge_normals[:,0])
        normal_yaw_lerp_start = jnp.repeat(normal_yaw, num_segments, axis=0)
        normal_yaw_lerp_end = jnp.repeat(jnp.concat([normal_yaw[1:], normal_yaw[:1]]), num_segments, axis=0)
        normal_yaw_lerp = lerp(normal_yaw_lerp_start, normal_yaw_lerp_end, lerp_weight)
        normal_yaw_lerp_deg = jnp.degrees(normal_yaw_lerp)
        if segment_normal_deg_range[1] > 180.0:
            in_segment_degs = ((segment_normal_deg_range[0] < normal_yaw_lerp_deg) & (normal_yaw_lerp_deg < 180.0+1e-3)) | ((-180.0-1e-3 < normal_yaw_lerp_deg) & (normal_yaw_lerp_deg < segment_normal_deg_range[1] - 360.0))
        else:
            in_segment_degs = (segment_normal_deg_range[0] < normal_yaw_lerp_deg) & (normal_yaw_lerp_deg < segment_normal_deg_range[1])
        in_segment_degs_ids = in_segment_degs.nonzero()[0]
        if in_segment_degs.shape[0] -1 in in_segment_degs_ids:
            discon_idx = jnp.diff(in_segment_degs_ids).argmax() + 1
            in_segment_degs_ids = jnp.concat([in_segment_degs_ids[discon_idx:], in_segment_degs_ids[:discon_idx]])

        self.eetrack_poss_b = eetrack_poss_b[in_segment_degs_ids]
        self.eetrack_quats_b = eetrack_quats_b[in_segment_degs_ids]
        self.eetrack_poses_b = jaxlie.SE3(jnp.concat([self.eetrack_quats_b, self.eetrack_poss_b], axis=1))
        self.number_of_subgoals = self.eetrack_poses_b.parameters().shape[0]

        # For debugging,
        # normal_vec = np.column_stack([self.eetrack_poss_b, self.eetrack_poss_b + 0.01*jaxlie.SO3(self.eetrack_quats_b).as_matrix()[:,:,0]])
        # normal_lerp = jnp.stack([jnp.cos(normal_yaw_lerp), jnp.sin(normal_yaw_lerp), jnp.zeros_like(normal_yaw_lerp)], axis=1)
        # normal_vec = np.column_stack([eetrack_poss_b, eetrack_poss_b + 0.01*normal_lerp])
        # trimesh.Scene([self.mesh, trimesh.PointCloud(eetrack_poss_b), trimesh.load_path(normal_vec.reshape((-1,2,3)))]).show()

    def get_welding_path(self, object_poses: jaxlie.SE3) -> jaxlie.SE3:
        num_object_poses = object_poses.parameters().shape[0]

        object_poses_batch = jaxlie.SE3(object_poses.parameters()[:,None].repeat(self.number_of_subgoals, axis=1))
        eetrack_poses_b_batch = jaxlie.SE3(self.eetrack_poses_b.parameters()[None].repeat(num_object_poses, axis=0))

        eetrack_poses_w_batch = object_poses_batch @ eetrack_poses_b_batch

        return eetrack_poses_w_batch


if __name__ == "__main__":
    weld_object = WeldObject(
        object_dir = "../weld_objects",
        object_name = "Circular_Body_Plate_135_100",
    )
    object_pose = jnp.array([[0.9123, 0.0, 0.0, 0.4095, 1.15, 2.4581, 0.29]]).repeat(10, axis=0)
    object_pose = jaxlie.SE3(object_pose)
    weld_line = weld_object.get_welding_path(object_pose)
