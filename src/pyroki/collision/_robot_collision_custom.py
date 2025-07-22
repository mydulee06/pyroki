from ._robot_collision import RobotCollision as BaseRobotCollision
import jax
import jax.numpy as jnp
from typing import Tuple
import yourdfpy
from ._geometry import Capsule
from typing import cast

class RobotCollision(BaseRobotCollision):
    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        user_ignore_pairs: Tuple[Tuple[str, str], ...] = (),
        ignore_immediate_adjacents: bool = True,
        exclude_links: Tuple[str, ...] = (),
    ):
        from .._robot_urdf_parser import RobotURDFParser
        # Re-load urdf with collision data if not already loaded.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        try:
            has_collision = any(link.collisions for link in urdf.link_map.values())
            if not has_collision:
                urdf = yourdfpy.URDF(
                    robot=urdf.robot,
                    filename_handler=filename_handler,
                    load_collision_meshes=True,
                )
        except Exception as e:
            import logging
            logging.warning(f"Could not reload URDF with collision meshes: {e}")
        _, link_info = RobotURDFParser.parse(urdf)
        link_name_list = link_info.names
        link_name_list = tuple([ln for ln in link_name_list if ln not in exclude_links])
        # Gather all collision meshes.
        cap_list = list[Capsule]()
        for link_name in link_name_list:
            cap_list.append(
                Capsule.from_trimesh(
                    BaseRobotCollision._get_trimesh_collision_geometries(urdf, link_name)
                )
            )
        capsules = cast(Capsule, jax.tree.map(lambda *args: jnp.stack(args), *cap_list))
        assert capsules.get_batch_axes() == (len(link_name_list),)
        active_idx_i, active_idx_j = RobotCollision._compute_active_pair_indices(
            link_names=link_name_list,
            urdf=urdf,
            user_ignore_pairs=user_ignore_pairs,
            ignore_immediate_adjacents=ignore_immediate_adjacents,
            exclude_links=exclude_links,
        )
        return RobotCollision(
            num_links=len(link_name_list),
            link_names=link_name_list,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            coll=capsules,
        )

    @staticmethod
    def _compute_active_pair_indices(
        link_names: Tuple[str, ...],
        urdf: yourdfpy.URDF,
        user_ignore_pairs: Tuple[Tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
        exclude_links: Tuple[str, ...] = (),
    ):
        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}
        ignore_matrix = jnp.zeros((num_links, num_links), dtype=bool)
        ignore_matrix = ignore_matrix.at[
            jnp.arange(num_links), jnp.arange(num_links)
        ].set(True)
        if ignore_immediate_adjacents:
            for joint in urdf.joint_map.values():
                parent_name = joint.parent
                child_name = joint.child
                if parent_name in link_name_to_idx and child_name in link_name_to_idx:
                    parent_idx = link_name_to_idx[parent_name]
                    child_idx = link_name_to_idx[child_name]
                    ignore_matrix = ignore_matrix.at[parent_idx, child_idx].set(True)
                    ignore_matrix = ignore_matrix.at[child_idx, parent_idx].set(True)
        for name1, name2 in user_ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_matrix = ignore_matrix.at[idx1, idx2].set(True)
                ignore_matrix = ignore_matrix.at[idx2, idx1].set(True)
        idx_i, idx_j = jnp.tril_indices(num_links, k=-1)
        should_check = ~ignore_matrix[idx_i, idx_j]
        active_i = idx_i[should_check]
        active_j = idx_j[should_check]
        return active_i, active_j 