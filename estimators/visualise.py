from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_3d_timeseries(
    time: list[datetime],
    x: list[float],
    y: list[float],
    z: list[float],
    title: str,
    x_truth: list[float] = None,
    y_truth: list[float] = None,
    z_truth: list[float] = None,
):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(
                    size=8,
                    cmax=39,
                    cmin=0,
                    color=[(t - time[0]).total_seconds() for t in time],
                    colorbar=dict(title="Seconds"),
                    colorscale="Viridis",
                ),
                mode="markers",
            )
        ],
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_truth,
            y=y_truth,
            z=z_truth,
            marker=dict(size=8, cmax=39, cmin=0, color="black"),
            mode="markers",
        )
    )
    fig.update_layout(
        scene=dict(yaxis_title="z", zaxis_title="y"),
        height=1200,
        width=1600,
        title_text=title,
    )
    fig.show()


def plot_2d_timeseries(
    time: list[datetime], x: list[float], y: list[float], z: list[float], title: str
):
    fig = make_subplots(rows=3, cols=1, subplot_titles=("X", "Y", "Z"))

    fig.add_trace(go.Scatter(x=time, y=x), row=1, col=1)

    fig.add_trace(go.Scatter(x=time, y=y), row=2, col=1)

    fig.add_trace(go.Scatter(x=time, y=z), row=3, col=1)

    fig.update_layout(height=1200, width=1600, title_text=title)
    fig.show()
