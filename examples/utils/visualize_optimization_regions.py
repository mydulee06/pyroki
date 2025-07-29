import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import plotly.express as px

optimization_data = [
    {"height": 0.46, "center": [-0.31134863, 0.29427807, 0.18421204], "range": [0.118, 0.050, 0.300], "success_rate": 0.986046511627907},
    {"height": 0.44, "center": [-0.32943383, 0.31302579, 0.0618972], "range": [0.150, 0.071, 0.300], "success_rate": 0.9977777777777778},
    {"height": 0.42, "center": [-0.30606729, 0.34168499, -0.0602559], "range": [0.150, 0.086, 0.300], "success_rate": 0.9979296066252588},
    {"height": 0.40, "center": [-0.3522652, 0.33930602, 0.01556581], "range": [0.150, 0.106, 0.300], "success_rate": 0.9982964224872232},
    {"height": 0.39, "center": [-0.28861955, 0.35248703, -0.15165929], "range": [0.150, 0.105, 0.300], "success_rate": 0.9967479674796748},
    {"height": 0.37, "center": [-0.35064208, 0.34662998, 0.13504088], "range": [0.150, 0.112, 0.300], "success_rate": 0.9971830985915493},
    {"height": 0.36, "center": [-0.27799724, 0.34397085, -0.03413953], "range": [0.150, 0.107, 0.300], "success_rate": 0.9985507246376811},
    {"height": 0.34, "center": [-0.29409094, 0.38023231, -0.19480077], "range": [0.150, 0.130, 0.300], "success_rate": 0.9986996098829649},
    {"height": 0.32, "center": [-0.23256217, 0.38154574, -0.45882671], "range": [0.150, 0.134, 0.300], "success_rate": 1.0},
    {"height": 0.30, "center": [-0.21333064, 0.37576616, -0.22281561], "range": [0.150, 0.119, 0.300], "success_rate": 1.0},
]

def create_rectangle_boundary(center_x, center_y, dx, dy):
    return {
        'x': [center_x - dx, center_x + dx, center_x + dx, center_x - dx, center_x - dx],
        'y': [center_y - dy, center_y - dy, center_y + dy, center_y + dy, center_y - dy]
    }

def create_3d_visualization():
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, data in enumerate(optimization_data):
        height = data["height"]
        center_x, center_y, center_yaw = data["center"]
        dx, dy, dyaw = data["range"]
        success_rate = data["success_rate"]
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter3d(
            x=[center_x],
            y=[center_y],
            z=[height],
            mode='markers',
            marker=dict(
                size=12,
                color=color,
                symbol='circle'
            ),
            name=f'h={height}m (SR={success_rate*100:.1f}%)',
            hovertemplate=(
                f'<b>Height: {height}m</b><br>'
                f'Center: ({center_x:.3f}, {center_y:.3f})<br>'
                f'Range: ±({dx:.3f}, {dy:.3f})<br>'
                f'Success Rate: {success_rate*100:.1f}%<br>'
                f'Yaw: {center_yaw:.3f} rad<extra></extra>'
            )
        ))
        
        boundary = create_rectangle_boundary(center_x, center_y, dx, dy)
        fig.add_trace(go.Scatter3d(
            x=boundary['x'],
            y=boundary['y'],
            z=[height] * 5,
            mode='lines',
            line=dict(color=color, width=6),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        arrow_length = 0.05
        arrow_x = center_x + arrow_length * np.cos(center_yaw)
        arrow_y = center_y + arrow_length * np.sin(center_yaw)
        
        fig.add_trace(go.Scatter3d(
            x=[center_x, arrow_x],
            y=[center_y, arrow_y],
            z=[height, height],
            mode='lines',
            line=dict(color='red', width=8),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(
            text='🎯 3D Optimization Regions by Height',
            font=dict(size=20),
            x=0.5
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Height (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

def create_trends_analysis():
    heights = [data["height"] for data in optimization_data]
    centers_x = [data["center"][0] for data in optimization_data]
    centers_y = [data["center"][1] for data in optimization_data]
    centers_yaw = [data["center"][2] for data in optimization_data]
    ranges_x = [data["range"][0] for data in optimization_data]
    ranges_y = [data["range"][1] for data in optimization_data]
    success_rates = [data["success_rate"] for data in optimization_data]
    
    fig = sp.make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Height vs X Center', 'Height vs Y Center', 'Height vs Yaw Center',
            'Height vs X Range', 'Height vs Y Range', 'Height vs Success Rate'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(go.Scatter(
        x=heights, y=centers_x,
        mode='lines+markers',
        name='X Center',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=heights, y=centers_y,
        mode='lines+markers',
        name='Y Center',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=heights, y=centers_yaw,
        mode='lines+markers',
        name='Yaw Center',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ), row=1, col=3)
    
    fig.add_trace(go.Scatter(
        x=heights, y=ranges_x,
        mode='lines+markers',
        name='X Range',
        line=dict(color='cyan', width=3),
        marker=dict(size=8)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=heights, y=ranges_y,
        mode='lines+markers',
        name='Y Range',
        line=dict(color='magenta', width=3),
        marker=dict(size=8)
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=heights, y=success_rates,
        mode='lines+markers',
        name='Success Rate',
        line=dict(color='black', width=3),
        marker=dict(size=8)
    ), row=2, col=3)
    
    fig.update_xaxes(title_text="Height (m)", row=1, col=1)
    fig.update_xaxes(title_text="Height (m)", row=1, col=2)
    fig.update_xaxes(title_text="Height (m)", row=1, col=3)
    fig.update_xaxes(title_text="Height (m)", row=2, col=1)
    fig.update_xaxes(title_text="Height (m)", row=2, col=2)
    fig.update_xaxes(title_text="Height (m)", row=2, col=3)
    
    fig.update_yaxes(title_text="X Center (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Center (m)", row=1, col=2)
    fig.update_yaxes(title_text="Yaw Center (rad)", row=1, col=3)
    fig.update_yaxes(title_text="X Range (±m)", row=2, col=1)
    fig.update_yaxes(title_text="Y Range (±m)", row=2, col=2)
    fig.update_yaxes(title_text="Success Rate", row=2, col=3)
    
    fig.update_layout(
        title=dict(
            text='📈 Optimization Trends Analysis',
            font=dict(size=20),
            x=0.5
        ),
        showlegend=False,
        width=1200,
        height=800
    )
    
    return fig

def create_optimal_path():
    fig = go.Figure()
    
    heights = np.array([data["height"] for data in optimization_data])
    centers_x = np.array([data["center"][0] for data in optimization_data])
    centers_y = np.array([data["center"][1] for data in optimization_data])
    
    fig.add_trace(go.Scatter3d(
        x=centers_x,
        y=centers_y,
        z=heights,
        mode='lines+markers',
        line=dict(color='red', width=8),
        marker=dict(
            size=10,
            color=heights,
            colorscale='Viridis',
            colorbar=dict(title='Height (m)')
        ),
        name='Optimal Path',
        hovertemplate=(
            '<b>Height: %{z}m</b><br>'
            'Position: (%{x:.3f}, %{y:.3f})<extra></extra>'
        )
    ))
    
    colors = px.colors.qualitative.Set3
    
    for i, data in enumerate(optimization_data):
        height = data["height"]
        center_x, center_y = data["center"][:2]
        dx, dy = data["range"][:2]
        color = colors[i % len(colors)]
        
        theta = np.linspace(0, 2*np.pi, 21)
        x_boundary = center_x + dx * np.cos(theta)
        y_boundary = center_y + dy * np.sin(theta)
        z_boundary = np.full_like(x_boundary, height)
        
        fig.add_trace(go.Scatter3d(
            x=x_boundary,
            y=y_boundary,
            z=z_boundary,
            mode='lines',
            line=dict(color=color, width=6),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(
            text='🔄 Optimal Path Through Different Heights',
            font=dict(size=20),
            x=0.5
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Height (m)',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            )
        ),
        width=1000,
        height=700
    )
    
    return fig

def create_combined_dashboard():
    fig_3d = create_3d_visualization()
    fig_trends = create_trends_analysis()
    fig_path = create_optimal_path()
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimization Regions 3D Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .plot-container {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .info-panel {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .stat-card {{
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.1);
            }}
            .stat-title {{
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Optimization Regions 3D Visualization</h1>
                <p>서로 다른 높이에서의 최적화된 영역 분석</p>
                <p>Generated with Python Plotly</p>
            </div>

            <div class="info-panel">
                <h3>📊 데이터 개요</h3>
                <p>이 시각화는 10개의 서로 다른 높이(0.30m~0.46m)에서 로봇의 최적화된 작업 영역을 보여줍니다.</p>
                <ul>
                    <li><strong>X, Y축:</strong> 로봇의 위치 좌표</li>
                    <li><strong>Z축 (Height):</strong> 로봇의 높이</li>
                    <li><strong>색상:</strong> 높이에 따른 구분</li>
                    <li><strong>빨간 화살표:</strong> 최적 Yaw 방향</li>
                </ul>
            </div>

            <div class="plot-container">
                <h3>🌟 Main 3D Visualization</h3>
                <div id="plot3d"></div>
            </div>

            <div class="plot-container">
                <h3>📈 Trend Analysis</h3>
                <div id="trends"></div>
            </div>

            <div class="plot-container">
                <h3>🔄 Optimal Path</h3>
                <div id="path"></div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">높이 범위</div>
                    <div>0.30m ~ 0.46m</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">성공률 범위</div>
                    <div>98.6% ~ 100%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">최고 성공률</div>
                    <div>100% (0.32m, 0.30m)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">X 좌표 범위</div>
                    <div>-0.35 ~ -0.21m</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Y 좌표 범위</div>
                    <div>0.29 ~ 0.38m</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">평균 X 범위</div>
                    <div>±0.14m</div>
                </div>
            </div>
        </div>

        <script>
            {plot3d_json}
            {trends_json}
            {path_json}
        </script>
    </body>
    </html>
    """
    
    plot3d_json = fig_3d.to_json()
    trends_json = fig_trends.to_json()
    path_json = fig_path.to_json()
    
    plot3d_script = f"Plotly.newPlot('plot3d', {plot3d_json}.data, {plot3d_json}.layout, {{responsive: true}});"
    trends_script = f"Plotly.newPlot('trends', {trends_json}.data, {trends_json}.layout, {{responsive: true}});"
    path_script = f"Plotly.newPlot('path', {path_json}.data, {path_json}.layout, {{responsive: true}});"
    
    return html_template.format(
        plot3d_json=plot3d_script,
        trends_json=trends_script,
        path_json=path_script
    )

def save_individual_plots():
    print("🔧 Creating individual visualizations...")
    
    fig_3d = create_3d_visualization()
    plot(fig_3d, filename='optimization_3d_main.html', auto_open=False)
    print("✅ Saved: optimization_3d_main.html")
    
    fig_trends = create_trends_analysis()
    plot(fig_trends, filename='optimization_trends.html', auto_open=False)
    print("✅ Saved: optimization_trends.html")
    
    fig_path = create_optimal_path()
    plot(fig_path, filename='optimization_path.html', auto_open=False)
    print("✅ Saved: optimization_path.html")

def main():
    print("🎯 Optimization Regions Visualization")
    print("=" * 50)
    
    print("\n1. Creating combined dashboard...")
    dashboard_html = create_combined_dashboard()
    
    with open('optimization_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print("✅ Saved: optimization_dashboard.html")
    
    print("\n2. Creating individual plots...")
    save_individual_plots()
    
    print("\n✅ All visualizations completed!")
    print("\n📄 Generated files:")
    print("- optimization_dashboard.html (통합 대시보드)")
    print("- optimization_3d_main.html (3D 메인 시각화)")
    print("- optimization_trends.html (트렌드 분석)")
    print("- optimization_path.html (최적 경로)")
    
    print("\n📊 Data Summary:")
    print("-" * 30)
    for data in optimization_data:
        print(f"Height: {data['height']}m, "
              f"Center: ({data['center'][0]:.3f}, {data['center'][1]:.3f}, {data['center'][2]:.3f}), "
              f"Range: ±({data['range'][0]:.3f}, {data['range'][1]:.3f}, {data['range'][2]:.3f}), "
              f"Success: {data['success_rate']:.3f}")

if __name__ == "__main__":
    main() 