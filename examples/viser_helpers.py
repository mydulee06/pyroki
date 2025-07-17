import viser
import numpy as np
from typing import Dict, Any, List, Tuple
import jaxlie
import jax.numpy as jnp
import pyroki as pk

def setup_viser_gui(server: viser.ViserServer, num_timesteps: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup Viser GUI controls."""
    gui_controls = {}
    
    # Playback controls
    gui_controls['playing'] = server.gui.add_checkbox("playing", True)
    gui_controls['timestep_slider'] = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    
    # Status and error display
    gui_controls['status_text'] = server.gui.add_text("status", "⏳ Please click 'Optimize!' to start")
    gui_controls['error_text'] = server.gui.add_text("error", "Waiting for optimization...")
    
    # Collision information
    gui_controls['collision_text'] = server.gui.add_text("collision", "⏳ Waiting for optimization")
    gui_controls['min_distance_text'] = server.gui.add_text("min_distance", "⏳ Waiting for optimization")
    
    # Collision visualization controls
    gui_controls['show_collision_meshes'] = server.gui.add_checkbox("show_collision_meshes", False)
    gui_controls['collision_threshold_slider'] = server.gui.add_slider("collision_threshold", 0.0, 0.2, 0.01, 0.1)
    
    return gui_controls


def setup_collision_visualization(
    server: viser.ViserServer, 
    robot_coll: pk.collision.RobotCollision
) -> Tuple[List, Dict[str, Any]]:
    """Setup collision visualization."""
    collision_mesh_handles = []
    collision_cache = {
        'active_pairs': robot_coll.active_idx_i,
        'link_names': robot_coll.link_names,
        'collision_distances': []
    }
    
    # Create collision mesh handles for each link
    for i, link_name in enumerate(robot_coll.link_names):
        if hasattr(robot_coll, 'link_meshes') and i < len(robot_coll.link_meshes):
            mesh = robot_coll.link_meshes[i]
            if mesh is not None:
                handle = server.scene.add_mesh_trimesh(
                    f"collision_{link_name}",
                    mesh,
                    visible=False,
                    color=(0.8, 0.8, 0.8, 0.1)
                )
                collision_mesh_handles.append(handle)
            else:
                collision_mesh_handles.append(None)
        else:
            collision_mesh_handles.append(None)
    
    return collision_mesh_handles, collision_cache


def update_collision_bodies(
    joints: jnp.ndarray,
    robot_coll: pk.collision.RobotCollision,
    collision_mesh_handles: List,
    collision_cache: Dict[str, Any],
    gui_controls: Dict[str, Any],
    robot: pk.Robot
):
    """Update collision body visualization."""
    # Compute collision distances
    collision_distances = robot_coll.compute_self_collision_distance(robot, joints)
    collision_cache['collision_distances'] = collision_distances
    
    # Find collision statistics
    min_distance = float('inf')
    collision_count = 0
    threshold = gui_controls['collision_threshold_slider'].value
    safety_margin = 0.05  # From config
    
    # Reset all collision meshes
    for handle in collision_mesh_handles:
        if handle:
            handle.visible = False
            handle.color = (0.8, 0.8, 0.8, 0.1)
    
    # Analyze collision pairs
    for i, (link1_idx, link2_idx) in enumerate(zip(robot_coll.active_idx_i, robot_coll.active_idx_j)):
        if i < len(collision_distances):
            distance = collision_distances[i]
            
            # Count violations
            if distance < safety_margin:
                collision_count += 1
            
            # Find minimum distance
            if distance < min_distance:
                min_distance = distance
            
            # Visualize collision pairs based on threshold
            if distance < threshold and gui_controls['show_collision_meshes'].value:
                if distance < safety_margin:
                    color = (1.0, 0.0, 0.0, 0.5)  # Red for violations
                else:
                    color = (1.0, 0.5, 0.0, 0.5)  # Orange for warnings
                
                # Show collision meshes
                if link1_idx < len(collision_mesh_handles) and collision_mesh_handles[link1_idx]:
                    collision_mesh_handles[link1_idx].visible = True
                    collision_mesh_handles[link1_idx].color = color
                
                if link2_idx < len(collision_mesh_handles) and collision_mesh_handles[link2_idx]:
                    collision_mesh_handles[link2_idx].visible = True
                    collision_mesh_handles[link2_idx].color = color
    
    # Update GUI text
    if min_distance < float('inf'):
        if collision_count > 0:
            gui_controls['collision_text'].value = f"🔴 Collisions: {collision_count}"
        else:
            gui_controls['collision_text'].value = "✅ No collisions"
        
        gui_controls['min_distance_text'].value = f"Min distance: {min_distance:.4f}m"
    else:
        gui_controls['collision_text'].value = "✅ No collisions"
        gui_controls['min_distance_text'].value = "Min distance: ∞"


def update_visualization(
    tstep: int,
    joints: jnp.ndarray,
    Ts_world_root: Tuple[jaxlie.SE3, ...],
    target_points: np.ndarray,
    target_colors: np.ndarray,
    position_errors: List[float],
    orientation_errors: List[float],
    min_distances: List[float],
    collision_violations: List[int],
    collision_distances: List[float],
    position_failed: bool,
    orientation_failed: bool,
    base_frame: viser.FrameHandle,
    urdf_vis: viser.extras.ViserUrdf,
    server: viser.ViserServer,
    gui_controls: Dict[str, Any],
    config: Dict[str, Any]
):
    """Update visualization with current timestep data."""
    
    # Update robot pose
    base_frame.wxyz = np.array(Ts_world_root[tstep].wxyz_xyz[:4])
    base_frame.position = np.array(Ts_world_root[tstep].wxyz_xyz[4:])
    urdf_vis.update_cfg(np.array(joints[tstep]))
    
    # Update target path visualization
    server.scene.add_point_cloud(
        "/target_path",
        target_points[gui_controls['timestep_slider'].value:gui_controls['timestep_slider'].value+1],
        target_colors[gui_controls['timestep_slider'].value:gui_controls['timestep_slider'].value+1],
        point_size=config['visualization']['point_size'],
    )
    
    # Update error display
    if tstep < len(position_errors) and tstep < len(orientation_errors):
        current_pos_error = position_errors[tstep]
        current_ori_error = orientation_errors[tstep]
        
        gui_controls['error_text'].value = (
            f"Position: {current_pos_error:.4f}m, "
            f"Orientation: {current_ori_error:.4f}rad"
        )
        
        # Update status based on errors
        if position_failed or orientation_failed:
            gui_controls['status_text'].value = "❌ Errors exceed tolerance"
        else:
            gui_controls['status_text'].value = "✅ All errors within tolerance"
    
    # Update timestep if playing
    if gui_controls['playing'].value:
        gui_controls['timestep_slider'].value = (gui_controls['timestep_slider'].value + 1) % len(joints)


 