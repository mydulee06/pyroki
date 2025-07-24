import os
import torch
import argparse
from pathlib import Path
from pprint import pprint
import numpy as np
from cmaes import CMA
from datetime import datetime
from functools import partial

# To disable futurewarning in torch.load
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_sample_mask_within_range(x, xyyaw_samples):
    # Ellipse range. Rectangular range also could be considered.
    xy_mask = ((xyyaw_samples[:,0] - x[0])/x[3])**2 + ((xyyaw_samples[:,1] - x[1])/x[4])**2 < 1
    yaw_mask = np.abs(xyyaw_samples[:,2] - x[2]) < x[5]
    mask = xy_mask & yaw_mask

    return mask

def compute_sr(x, xyyaw_samples, success):
    mask = compute_sample_mask_within_range(x, xyyaw_samples)

    num_total = mask.sum()
    num_success = success[mask].sum()

    sr = num_success / num_total if num_total > 0.0 else 0.0
    return sr, num_success, num_total

def cost_fn(x, xyyaw_samples, success, cmaes_cfg, cost_cfg):
    sr, num_success, num_total = compute_sr(x, xyyaw_samples, success)

    total_cost = 0
    info = {}
    # Cost for success rate
    if cost_cfg["sr"]["enabled"]:
        sr_cost = cost_cfg["sr"]["weight"] * (1 - sr) # or 1 / (sr + 1e-6)
        total_cost += sr_cost
        info["sr"] = sr_cost
    # Cost for num_sample
    if cost_cfg["num_sample"]["enabled"]:
        num_sample_cost = cost_cfg["num_sample"]["weight"] * (1 / num_total)
        total_cost += num_sample_cost
        info["num_sample"] = num_sample_cost

    bounds = cmaes_cfg["bounds"]
    # Cost for dx
    if cost_cfg["dx"]["enabled"]:
        dx_cost = cost_cfg["dx"]["weight"] * (bounds[3,1] - x[3])
        total_cost += dx_cost
        info["dx"] = dx_cost
    # Cost for dy
    if cost_cfg["dy"]["enabled"]:
        dy_cost = cost_cfg["dy"]["weight"] * (bounds[4,1] - x[4])
        total_cost += dy_cost
        info["dy"] = dy_cost
    # Cost for dyaw
    if cost_cfg["dyaw"]["enabled"]:
        dyaw_cost = cost_cfg["dyaw"]["weight"] * (bounds[5,1] - x[5])
        total_cost += dyaw_cost
        info["dyaw"] = dyaw_cost

    return total_cost, info

def wrap_title(title, max_chars=250):
    words = title.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return "<br>".join(lines)

def animate_cmaes(save_dir, prefix, xyyaw_samples, success, x_traj, cost_traj, per_cost_traj, cmaes_result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import json

    def cylinder_plot(pos, x_radius, y_radius, height, resolution=50, **kwargs):
        theta = np.linspace(0, 2*np.pi, resolution)
        z = np.linspace(-height/2, height/2, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)

        # Convert to Cartesian coordinates
        x = pos[0] + x_radius * np.cos(theta_grid)
        y = pos[1] + y_radius * np.sin(theta_grid)
        z = pos[2] + z_grid

        return go.Surface(x=x, y=y, z=z, colorscale=[[0, 'blue'], [1, 'blue']], opacity=0.5, showscale=False, **kwargs)

    x, y, yaw = xyyaw_samples[:,0], xyyaw_samples[:,1], xyyaw_samples[:,2]

    if len(x_traj) < 100:
        interest_ids = np.arange(len(x_traj))
    else:
        # If too large x_traj, reduce it.
        after_step = int(np.round((len(x_traj) - 20) / 100))
        interest_ids = np.array(list(range(20)) + list(range(20, len(x_traj), after_step)))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
    )

    fig.add_trace(go.Scatter3d(
        x=x[success], y=y[success], z=yaw[success],
        mode='markers',
        marker=dict(size=4, color='green'),
        name=f"Success",
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=x[~success], y=y[~success], z=yaw[~success],
        mode='markers',
        marker=dict(size=4, color='red'),
        name=f"Fail",
    ), row=1, col=1)

    bit_larger_x = cmaes_result["x"].copy()
    bit_larger_x[3:5] += 0.01
    mask = compute_sample_mask_within_range(bit_larger_x, xyyaw_samples)

    success_within_range = success & mask
    fig.add_trace(go.Scatter3d(
        x=x[success_within_range], y=y[success_within_range], z=yaw[success_within_range],
        mode='markers',
        marker=dict(size=4, color='green'),
        name=f"Success within range",
    ), row=1, col=1)

    fail_within_range = (~success) & mask
    fig.add_trace(go.Scatter3d(
        x=x[fail_within_range], y=y[fail_within_range], z=yaw[fail_within_range],
        mode='markers',
        marker=dict(size=4, color='red'),
        name=f"Fail within range",
    ), row=1, col=1)

    for i, id in enumerate(interest_ids):
        pos = x_traj[id, :3]
        x_radius = x_traj[id, 3]
        y_radius = x_traj[id, 4]
        height = 2*x_traj[id, 5] # height = 2*dyaw
        fig.add_trace(
            cylinder_plot(
                pos = pos,
                x_radius = x_radius,
                y_radius = y_radius,
                height = height,
                visible = (i == len(x_traj) + 1),
                name = f"Cost: {cost_traj[id]}",
            )
        )

    fig.add_trace(go.Scatter(
        y=cost_traj,
        # mode='markers',
        # marker=dict(size=10, color=color[mask]),
        name=f'total cost',
    ), row=1, col=2)
    for cost_name, cost_value_traj in per_cost_traj.items():
        fig.add_trace(go.Scatter(
            y=cost_value_traj,
            # mode='markers',
            # marker=dict(size=10, color=color[mask]),
            name=f'{cost_name} cost',
        ), row=1, col=2)

    steps = []
    for i, id in enumerate(interest_ids):
        visibility = [True, True, True, True] + [j == i for j in range(len(interest_ids))] + len(per_cost_traj) * [True] # 2D scatter
        steps.append(dict(
            method='update',
            args=[{'visible': visibility}],
            label=f"{id}"
        ))

    cfg = cmaes_result["cmaes_cfg"] | cmaes_result["cost_cfg"]
    cfg["final_x_y_yaw"] = cmaes_result["x"][:3]
    cfg["final_dx_dy_dyaw"] = cmaes_result["x"][3:]
    cfg["success_rate"] = f"{cmaes_result['success_rate']} ({cmaes_result['num_success']}/{cmaes_result['num_total']})"
    cfg_list = {k: np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v for k, v in cfg.items()}
    cfg_str = json.dumps(cfg_list)
    # Add slider layout
    fig.update_layout(
        title_text = wrap_title(cfg_str),
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Iteration : "},
            pad={"t": 50},
            steps=steps
        )],
        scene=dict(
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            zaxis_title='yaw (rad)',
            aspectmode='cube',
            xaxis=dict(range=[x.min()-0.02, x.max()+0.02]),
            yaxis=dict(range=[y.min()-0.02, y.max()+0.02]),
            zaxis=dict(range=[yaw.min()-0.02, yaw.max()+0.02]),
        ),
        xaxis_title='Iteration',
        yaxis_title='Cost',
        xaxis_range=[-5, len(cost_traj)+5],
        showlegend=True,
        height=1000,
        width=2000,
    )

    save_path = os.path.abspath(os.path.join(save_dir, f"{prefix}_cmaes_animation.html"))
    fig.write_html(save_path)
    print(f"Animation saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="files", help="Log directory")
    parser.add_argument("--exp_prefix", type=str, help="Prefix of experiment name.")
    parser.add_argument("--algo", type=str, default="grid", choices=["grid", "cmaes"], help="Device for torch")
    parser.add_argument("--dxy", type=float, default=0.1, help="xy bound for search")
    parser.add_argument("--dyaw", type=float, default=0.1, help="yaw bound for search")
    parser.add_argument("--num_grid", type=int, default=50, help="Number of grid")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device for torch")
    parser.add_argument("--max_dx", type=float, default=0.15, help="Maximum dx bound for CMA-ES")
    parser.add_argument("--max_dy", type=float, default=0.15, help="Maximum dx bound for CMA-ES")
    parser.add_argument("--max_dyaw", type=float, default=0.3, help="Maximum dyaw bound for CMA-ES")
    parser.add_argument("--verbose", action="store_true", default=False, help="Set verbose for CMA-ES")
    parser.add_argument("--save_cmaes_result", action="store_true", default=False, help="Save result of CMA-ES optimization.")
    parser.add_argument("--animate", action="store_true", default=False, help="Animate CMA-ES optimization.")
    args = parser.parse_args()

    device = args.device
    dxy = args.dxy
    dyaw = args.dyaw
    num_grid = args.num_grid
    
    exp_dir = Path(args.log_dir, args.exp_prefix)
    exp_paths = exp_dir.glob("*")

    for exp_path in exp_paths:
        save_path = exp_path / "best_base_pose.pt"
        if save_path.exists() and not "seed_42_weld_object_type_Rect_Plate_125_weld_object_z_0.29_0.29_x_-0.70_-0.10_y_-0.15_0.45_yaw_0.20_1.20_h_0.37_0.37_segment_degs_-1.00_1.00_appr_deg_45.0" in exp_path.as_posix():
            continue
        data_path = exp_path / "batch_eetrack_results_inverse.pt"

        # Try to load as torch file first, then as numpy file
        try:
            data = torch.load(data_path)
            print(f"Loaded {data_path} as torch file")
        except:
            print(f"Failed to load {data_path} as torch file")
            # Load as numpy file
            try:
                data = np.load(data_path, allow_pickle=True)
                data = {k: torch.tensor(v, device=device) if isinstance(v, np.ndarray) else v 
                        for k, v in data.items()}
                print(f"Loaded {data_path} as numpy file")
            except:
                print(f"Failed to load {data_path} as numpy file")
                continue

        xyyaw_range = data["sample_range"]
        xyyaw_samples = data["xyyaw_samples"]
        success = data["success"]

        # The exp with ellipse range
        if "rx" in xyyaw_range:
            continue

        if args.algo == "grid":
            x_min, x_max = xyyaw_range["x"]
            x_min += args.dxy
            x_max -= args.dxy
            y_min, y_max = xyyaw_range["y"]
            y_min += args.dxy
            y_max -= args.dxy
            yaw_min, yaw_max = xyyaw_range["yaw"]
            yaw_min += args.dyaw
            yaw_max -= args.dyaw

            x_grid = torch.linspace(x_min, x_max, num_grid+1, device=device)
            y_grid = torch.linspace(y_min, y_max, num_grid+1, device=device)
            yaw_grid = torch.linspace(yaw_min, yaw_max, num_grid+1, device=device)

            grid = torch.stack(torch.meshgrid(x_grid, y_grid, yaw_grid), dim=-1).reshape(-1,3)
            dist = xyyaw_samples[None] - grid[:,None]
            xy_dist = dist[...,:2].norm(dim=-1)
            yaw_dist = dist[...,2].abs()

            grid_mask = (xy_dist < dxy) & (yaw_dist < dyaw)
            grid_success_mask = success.repeat(grid.shape[0],1) & grid_mask
            grid_num_success = grid_success_mask.sum(dim=-1)
            grid_num_total = grid_mask.sum(dim=-1)
            grid_success_rate = grid_num_success / grid_num_total
            grid_success_rate[grid_success_rate.isnan()] = 0.0
            ids = (grid_success_rate == grid_success_rate.max()).nonzero().flatten()
            ids = ids[grid_num_success[ids] == grid_num_success[ids].max()]
            robust_xyyaws = grid[ids].tolist()
            robust_xyyaw_sr = grid_success_rate[ids][0].item()
            num_success = grid_success_mask.sum(dim=-1)[ids][0].item()
            num_total = grid_mask.sum(dim=-1)[ids][0].item()

        elif args.algo == "cmaes":
            x_range = xyyaw_range['x']
            y_range = xyyaw_range['y']
            yaw_range = xyyaw_range['yaw']
            xyyaw_samples = xyyaw_samples.cpu().numpy()
            success = success.cpu().numpy()

            cmaes_cfg = {
                "seed": 42,
                "mean": np.array([
                    np.mean(x_range),
                    np.mean(y_range),
                    np.mean(yaw_range),
                    args.max_dx,
                    args.max_dy,
                    args.max_dyaw,
                ]),
                "sigma": 0.5,
                "bounds": np.array([
                    np.array([x_range[0]+args.max_dx, x_range[1]-args.max_dx]),
                    np.array([y_range[0]+args.max_dy, y_range[1]-args.max_dy]),
                    np.array([yaw_range[0]+args.max_dyaw, yaw_range[1]-args.max_dyaw]),
                    [0.05, args.max_dx],
                    [0.05, args.max_dy],
                    [0.05, args.max_dyaw],
                ]),
                "population_size": 32,
                "lr_adapt": False,
            }
            cost_cfg = {
                "sr": {
                    "enabled": True,
                    "weight": 1.0,
                },
                "num_sample": {
                    "enabled": True,
                    "weight": 2.5,
                },
                "dx": {
                    "enabled": True,
                    "weight": 0.2,
                },
                "dy": {
                    "enabled": True,
                    "weight": 0.1,
                },
                "dyaw": {
                    "enabled": True,
                    "weight": 0.1,
                },
            }

            wrapped_cost_fn = partial(
                cost_fn,
                xyyaw_samples=xyyaw_samples,
                success=success,
                cmaes_cfg=cmaes_cfg,
                cost_cfg=cost_cfg,
            )

            optimizer = CMA(
                **cmaes_cfg
            )
            if args.verbose:
                print(" evals    cost")
                print("======  ==========")

            evals = 0
            x_traj = [cmaes_cfg["mean"]]
            cost_traj = [wrapped_cost_fn(cmaes_cfg["mean"])[0]]
            per_cost_traj = {cost_name: [cost_value] for cost_name, cost_value in wrapped_cost_fn(cmaes_cfg["mean"])[1].items()}

            while True:
                solutions = []
                for _ in range(optimizer.population_size):
                    x = optimizer.ask()
                    cost, cost_info = wrapped_cost_fn(x)
                    evals += 1
                    solutions.append((x, cost))
                    if evals % 50 == 0:
                        x_traj.append(x)
                        cost_traj.append(cost)
                        for cost_name, cost_value in cost_info.items():
                            per_cost_traj[cost_name].append(cost_value)
                        if args.verbose:
                            print(f"{evals:5d}  {cost:10.5f}")
                optimizer.tell(solutions)

                if optimizer.should_stop():
                    break

            robust_xyyaws = x[:3]
            dx = x[3]
            dy = x[4]
            dyaw = x[5]
            robust_xyyaw_sr, num_success, num_total  = compute_sr(x, xyyaw_samples, success)

            x_traj = np.stack(x_traj)
            cost_traj = np.stack(cost_traj)
            date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            cmaes_dir = exp_path / "cmaes"
            cmaes_dir.mkdir(exist_ok=True)

            cmaes_result = {
                "cmaes_cfg": cmaes_cfg,
                "cost_cfg": cost_cfg,
                "x_traj": x_traj,
                "cost_traj": cost_traj,
                "x": x,
                "success_rate": robust_xyyaw_sr,
                "num_success": num_success,
                "num_total": num_total,
            }
            if args.save_cmaes_result:
                cmaes_save_path = cmaes_dir / f"{date_time}_cames_result.npz"
                np.savez(cmaes_save_path, **cmaes_result)
                print(f"CMA-ES result is saved to {cmaes_save_path}")

            if args.animate:
                animate_cmaes(
                    cmaes_dir,
                    date_time,
                    xyyaw_samples,
                    success,
                    x_traj,
                    cost_traj,
                    per_cost_traj,
                    cmaes_result,
                )

        pprint(f"The most succeessful x, y, yaw: {robust_xyyaws}")
        pprint(f"The most succeessful dx: {dx:.3f}, dy: {dy:.3f}, dyaw: {dyaw:.3f}")
        pprint(f"The most greatest success rate: {robust_xyyaw_sr} ({num_success}/{num_total})")
        save_data = {
            "xyyaws": robust_xyyaws,
            "dxy": dxy,
            "dyaw": dyaw,
            "success_rate": robust_xyyaw_sr,
            "num_success": num_success,
            "num_total": num_total,
        }
        torch.save(save_data, save_path)

if __name__=="__main__":
    main()