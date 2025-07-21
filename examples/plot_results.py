import json
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

# 결과 파일 경로
RESULTS_PATH = "batch_eetrack_results.json"
HTML_PATH = "batch_eetrack_results_plot.html"

# 결과 로드
def load_results(path):
    with open(path, "r") as f:
        results = json.load(f)
    return results

def main():
    results = load_results(RESULTS_PATH)
    # x, y, yaw, success 추출
    x = np.array([r['sampled_x'] for r in results])
    y = np.array([r['sampled_y'] for r in results])
    yaw = np.array([r['sampled_yaw'] for r in results])
    success = np.array([r['success'] for r in results])

    # 성공/실패 분리
    x_s, y_s, yaw_s = x[success], y[success], yaw[success]
    x_f, y_f, yaw_f = x[~success], y[~success], yaw[~success]

    trace_success = go.Scatter3d(
        x=x_s, y=y_s, z=yaw_s,
        mode='markers',
        marker=dict(size=5, color='green'),
        name='Success'
    )
    trace_fail = go.Scatter3d(
        x=x_f, y=y_f, z=yaw_f,
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Fail'
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='yaw (rad)'
        ),
        title='Batch EETRACK Optimization Results',
        legend=dict(x=0.8, y=0.9)
    )

    fig = go.Figure(data=[trace_success, trace_fail], layout=layout)
    plot(fig, filename=HTML_PATH, auto_open=False)
    print(f"Saved interactive plot to {HTML_PATH}")

if __name__ == "__main__":
    main() 