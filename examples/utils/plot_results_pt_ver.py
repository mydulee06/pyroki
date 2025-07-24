#!/usr/bin/env python3
"""
Plot PT Results - Interactive 3D visualization of PT file data
Supports both torch.load and fallback loading methods
"""

import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.express as px
import torch
import argparse
from pathlib import Path
import sys

def load_pt_results(pt_path: str):
    """
    Load results from PT file with fallback methods
    
    Args:
        pt_path: Path to PT file
        
    Returns:
        dict: Data dictionary with xyyaw_samples, success, etc.
    """
    print(f"📂 Loading PT file: {pt_path}")
    
    if not Path(pt_path).exists():
        raise FileNotFoundError(f"PT file not found: {pt_path}")
    
    # Method 1: Try torch.load with weights_only=False for compatibility
    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        print(f"✅ Loaded successfully with torch.load (weights_only=False)")
        
        # Convert torch tensors to numpy if needed
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.numpy()
                
        return data
        
    except Exception as e:
        print(f"⚠️  torch.load failed: {e}")
        print(f"🔄 Trying alternative loading methods...")
    
    # Method 2: Try torch.load with weights_only=True
    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        print(f"✅ Loaded successfully with torch.load (weights_only=True)")
        
        # Convert torch tensors to numpy if needed
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.numpy()
                
        return data
        
    except Exception as e:
        print(f"⚠️  torch.load (weights_only=True) failed: {e}")
    
    # Method 3: Try numpy load (for compatibility)
    try:
        np_data = np.load(pt_path, allow_pickle=True)
        print(f"✅ Loaded with np.load (compatibility mode)")
        
        if hasattr(np_data, 'item'):
            # If it's a numpy array containing a dict
            data = np_data.item()
        elif hasattr(np_data, 'files'):
            # If it's an npz file
            data = dict(np_data)
        else:
            data = np_data
            
        return data
        
    except Exception as e:
        print(f"❌ np.load also failed: {e}")
        raise RuntimeError(f"Could not load PT file with any method")

def analyze_data_structure(data):
    """
    Analyze and print the structure of loaded data
    """
    print(f"\n🔍 Data Structure Analysis:")
    print(f"   Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: numpy array, shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, torch.Tensor):
                print(f"   {key}: torch tensor, shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"   {key}: {type(value).__name__}, length={len(value)}")
            else:
                print(f"   {key}: {type(value).__name__} = {value}")
    elif hasattr(data, 'files'):
        print(f"   NPZ files: {data.files}")
        for file in data.files:
            arr = data[file]
            print(f"   {file}: shape={arr.shape}, dtype={arr.dtype}")
    else:
        print(f"   Value: {data}")

def extract_samples_and_success(data):
    """
    Extract x, y, yaw samples and success data from various data formats
    
    Returns:
        tuple: (xyyaw_samples, success) where:
               - xyyaw_samples is np.array of shape (N, 3)
               - success is np.array of shape (N,) with boolean values
    """
    
    # Try different possible key names and structures
    possible_sample_keys = ['xyyaw_samples', 'samples', 'positions', 'poses', 'x_y_yaw']
    possible_success_keys = ['success', 'successful', 'results', 'is_success']
    
    samples = None
    success_data = None
    
    # Look for sample data
    for key in possible_sample_keys:
        if key in data:
            samples = data[key]
            print(f"✅ Found samples data in key: '{key}'")
            break
    
    # Look for success data
    for key in possible_success_keys:
        if key in data:
            success_data = data[key]
            print(f"✅ Found success data in key: '{key}'")
            break
    
    # If we didn't find standard keys, try to infer from data structure
    if samples is None:
        # Try to find any 3D array that could be x, y, yaw
        for key, value in data.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                if hasattr(value, 'shape') and len(value.shape) == 2 and value.shape[1] == 3:
                    samples = value
                    print(f"✅ Inferred samples from key '{key}' (shape: {value.shape})")
                    break
    
    # If still no samples, try to construct from separate x, y, yaw
    if samples is None:
        x_data = data.get('x', data.get('sampled_x', None))
        y_data = data.get('y', data.get('sampled_y', None))
        yaw_data = data.get('yaw', data.get('sampled_yaw', None))
        
        if x_data is not None and y_data is not None and yaw_data is not None:
            samples = np.column_stack([x_data, y_data, yaw_data])
            print(f"✅ Constructed samples from separate x, y, yaw arrays")
    
    # If still no success data, try to infer or create dummy data
    if success_data is None:
        if samples is not None:
            # Create dummy success data (all True for visualization)
            success_data = np.ones(len(samples), dtype=bool)
            print(f"⚠️  No success data found, creating dummy success array (all True)")
        else:
            # Try to find any boolean array
            for key, value in data.items():
                if isinstance(value, (np.ndarray, torch.Tensor)) and hasattr(value, 'dtype'):
                    if value.dtype == bool or value.dtype == np.bool_:
                        success_data = value
                        print(f"✅ Inferred success data from key '{key}'")
                        break
    
    if samples is None:
        available_keys = list(data.keys()) if isinstance(data, dict) else "Not a dictionary"
        raise ValueError(f"Could not find sample data (x, y, yaw). Available keys: {available_keys}")
    
    # Convert to numpy arrays
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    if isinstance(success_data, torch.Tensor):
        success_data = success_data.numpy()
    
    # Ensure correct shapes
    if len(samples.shape) == 1:
        # If it's 1D, try to reshape to (N, 3)
        if len(samples) % 3 == 0:
            samples = samples.reshape(-1, 3)
        else:
            raise ValueError(f"Cannot reshape 1D array of length {len(samples)} to (N, 3)")
    
    if samples.shape[1] != 3:
        raise ValueError(f"Expected samples to have 3 columns (x, y, yaw), got {samples.shape[1]}")
    
    if success_data is None:
        success_data = np.ones(len(samples), dtype=bool)
    
    if len(success_data) != len(samples):
        print(f"⚠️  Warning: samples length ({len(samples)}) != success length ({len(success_data)})")
        # Trim to shorter length
        min_len = min(len(samples), len(success_data))
        samples = samples[:min_len]
        success_data = success_data[:min_len]
    
    return samples, success_data.astype(bool)
    """
    Create interactive 3D scatter plot from PT data
    
    Args:
        data: Data dictionary from PT file
        output_path: Output HTML file path
        show_stats: Whether to show statistics in console
    """
    
    # Extract data
    xyyaw_samples = data['xyyaw_samples']
    success = data['success'].astype(bool)
    
    x, y, yaw = xyyaw_samples[:, 0], xyyaw_samples[:, 1], xyyaw_samples[:, 2]
    
    # Split successful and failed samples
    x_success, y_success, yaw_success = x[success], y[success], yaw[success]
    x_fail, y_fail, yaw_fail = x[~success], y[~success], yaw[~success]
    
    if show_stats:
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(x):,}")
        print(f"   Successful: {success.sum():,} ({success.mean():.2%})")
        print(f"   Failed: {(~success).sum():,} ({(~success).mean():.2%})")
        print(f"\n📐 Sample Ranges:")
        print(f"   X: [{x.min():.4f}, {x.max():.4f}] (range: {x.max()-x.min():.4f})")
        print(f"   Y: [{y.min():.4f}, {y.max():.4f}] (range: {y.max()-y.min():.4f})")
        print(f"   Yaw: [{yaw.min():.4f}, {yaw.max():.4f}] (range: {yaw.max()-yaw.min():.4f})")
    
    # Create traces
    traces = []
    
    if len(x_success) > 0:
        trace_success = go.Scatter3d(
            x=x_success, 
            y=y_success, 
            z=yaw_success,
            mode='markers',
            marker=dict(
                size=3,
                color='green',
                opacity=0.8,
                symbol='circle'
            ),
            name=f'Success ({len(x_success):,})',
            hovertemplate='<b>Success</b><br>' +
                         'X: %{x:.4f}<br>' +
                         'Y: %{y:.4f}<br>' +
                         'Yaw: %{z:.4f}<br>' +
                         '<extra></extra>'
        )
        traces.append(trace_success)
    
    if len(x_fail) > 0:
        trace_fail = go.Scatter3d(
            x=x_fail, 
            y=y_fail, 
            z=yaw_fail,
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.6,
                symbol='circle'
            ),
            name=f'Failure ({len(x_fail):,})',
            hovertemplate='<b>Failure</b><br>' +
                         'X: %{x:.4f}<br>' +
                         'Y: %{y:.4f}<br>' +
                         'Yaw: %{z:.4f}<br>' +
                         '<extra></extra>'
        )
        traces.append(trace_fail)
    
    # Create layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X Position (m)', showgrid=True),
            yaxis=dict(title='Y Position (m)', showgrid=True),
            zaxis=dict(title='Yaw (rad)', showgrid=True),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title={
            'text': f'3D Visualization: {data.get("num_samples", len(x)):,} Samples '
                   f'({data.get("success_rate", success.mean()):.1%} Success Rate)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        width=1000,
        height=800
    )
    
    # Create figure
    fig = go.Figure(data=traces, layout=layout)
    
    # Save to file
    if output_path is None:
        output_path = "pt_results_3d_plot.html"
    
    plot(fig, filename=output_path, auto_open=False)
    print(f"✅ Saved interactive 3D plot to {output_path}")
    
    return fig

def create_2d_projections(data: dict, output_path: str = None):
    """
    Create 2D projection plots (XY, X-Yaw, Y-Yaw)
    """
    # Extract data with flexible parsing
    xyyaw_samples, success = extract_samples_and_success(data)
    
    x, y, yaw = xyyaw_samples[:, 0], xyyaw_samples[:, 1], xyyaw_samples[:, 2]
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('XY Projection', 'X-Yaw Projection', 'Y-Yaw Projection', 'Success Distribution'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # XY Projection
    fig.add_trace(
        go.Scatter(x=x[~success], y=y[~success], mode='markers',
                  marker=dict(size=2, color='red', opacity=0.6),
                  name='Failure', showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x[success], y=y[success], mode='markers',
                  marker=dict(size=3, color='green', opacity=0.8),
                  name='Success', showlegend=True),
        row=1, col=1
    )
    
    # X-Yaw Projection
    fig.add_trace(
        go.Scatter(x=x[~success], y=yaw[~success], mode='markers',
                  marker=dict(size=2, color='red', opacity=0.6),
                  name='Failure', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x[success], y=yaw[success], mode='markers',
                  marker=dict(size=3, color='green', opacity=0.8),
                  name='Success', showlegend=False),
        row=1, col=2
    )
    
    # Y-Yaw Projection
    fig.add_trace(
        go.Scatter(x=y[~success], y=yaw[~success], mode='markers',
                  marker=dict(size=2, color='red', opacity=0.6),
                  name='Failure', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=y[success], y=yaw[success], mode='markers',
                  marker=dict(size=3, color='green', opacity=0.8),
                  name='Success', showlegend=False),
        row=2, col=1
    )
    
    # Success Distribution
    fig.add_trace(
        go.Bar(x=['Failure', 'Success'], 
               y=[(~success).sum(), success.sum()],
               marker_color=['red', 'green'],
               name='Count', showlegend=False),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="X (m)", row=1, col=2)
    fig.update_yaxes(title_text="Yaw (rad)", row=1, col=2)
    fig.update_xaxes(title_text="Y (m)", row=2, col=1)
    fig.update_yaxes(title_text="Yaw (rad)", row=2, col=1)
    fig.update_xaxes(title_text="Result", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(
        title=f'2D Projections: {data.get("num_samples", len(x)):,} Samples',
        height=800,
        width=1200
    )
    
    if output_path is None:
        output_path = "pt_results_2d_projections.html"
    
    plot(fig, filename=output_path, auto_open=False)
    print(f"✅ Saved 2D projections to {output_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Create interactive plots from PT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", required=True,
                       help="Input PT file path")
    parser.add_argument("--output-3d", "-o3",
                       help="Output HTML file for 3D plot (default: pt_results_3d_plot.html)")
    parser.add_argument("--output-2d", "-o2",
                       help="Output HTML file for 2D projections (default: pt_results_2d_projections.html)")
    parser.add_argument("--plot-type", choices=['3d', '2d', 'both'], default='both',
                       help="Type of plots to generate")
    parser.add_argument("--no-stats", action='store_true',
                       help="Don't show statistics in console")
    
    args = parser.parse_args()
    
    try:
        # Load PT file
        data = load_pt_results(args.input)
        
        # Generate plots based on user choice
        if args.plot_type in ['3d', 'both']:
            print(f"\n🎨 Generating 3D scatter plot...")
            create_3d_scatter_plot(data, args.output_3d, not args.no_stats)
        
        if args.plot_type in ['2d', 'both']:
            print(f"\n🎨 Generating 2D projection plots...")
            create_2d_projections(data, args.output_2d)
        
        print(f"\n🎉 Plot generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()