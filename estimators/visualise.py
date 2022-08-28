from datetime import datetime

import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots


def plot_3d_timeseries(
    time: list[datetime],
    title: str,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    x_true: NDArray = None,
    y_true: NDArray = None,
    z_true: NDArray = None,
    scene: dict = None,
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
            x=x_true,
            y=y_true,
            z=z_true,
            marker=dict(
                size=8,
                cmax=39,
                cmin=0,
                color="Black",
            ),
        )
    )

    fig.update_layout(
        height=1200, width=1600, title_text=title, title_x=0.5, scene=scene
    )
    fig.show()


def plot_2d_timeseries(
    time: list[datetime], x: NDArray, y: NDArray, z: NDArray, title: str
):
    fig = make_subplots(rows=3, cols=1, subplot_titles=("X", "Y", "Z"))

    fig.add_trace(go.Scatter(x=time, y=x), row=1, col=1)

    fig.add_trace(go.Scatter(x=time, y=y), row=2, col=1)

    fig.add_trace(go.Scatter(x=time, y=z), row=3, col=1)

    fig.update_layout(height=1200, width=1600, title_text=title)
    fig.show()


def plot_uncertainty_timeseries(
    time: list[datetime], state: NDArray, uncertainty: NDArray, title: str
):
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Value", "Uncertainty"))

    fig.add_trace(go.Scatter(x=time, y=state), row=1, col=1)

    fig.add_trace(go.Scatter(x=time, y=uncertainty), row=2, col=1)

    fig.update_layout(height=1200, width=1600, title_text=title, title_x=0.5)
    fig.show()
