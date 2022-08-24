"""
TODO:
    - extend to EKF
"""
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import integrate
from scipy.spatial.transform import Rotation as Rot
from utils import generate_environment

from estimators.kalman_filter import KalmanFilter
from estimators.sensors import (
    Accelerometer,
    AccelerometerUncalibrated,
    Gyroscope,
    GyroscopeUncalibrated,
    Magnetometer,
    MagnetometerUncalibrated,
    Measurement,
    RotationVector,
    Waypoint,
)

ABS_PATH = Path(__file__).parent.resolve()

G = 9.81  # gravitational acceleration (m/s^2)


def plot_3d_timeseries(
    time: list[datetime],
    x: list[float],
    y: list[float],
    z: list[float],
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
        ]
    )
    fig.show()


def generate_control_input(
    time: datetime, accel: Measurement, gyro: Measurement
) -> np.array:
    # https://escholarship.org/content/qt5rs5t0sf/qt5rs5t0sf_noSplash_e1588dedf177d86a3652374bc997314f.pdf
    r_est = np.arctan2(accel.y, accel.z)
    p_est = np.arcsin(accel.x / G)
    # https://liqul.github.io/blog/assets/rotation.pdf
    R = np.array(
        [
            [1, np.sin(p_est) * np.tan(r_est), np.cos(p_est) * np.tan(r_est)],
            [0, np.cos(r_est), -np.sin(r_est)],
            [0, np.sin(r_est) / np.cos(p_est), np.cos(r_est) / np.cos(p_est)],
        ]
    )
    # control vector is  euler angle velocities of the IMU in world frame
    # https://calhoun.nps.edu/bitstream/handle/10945/34427/McGhee_bachmann_zyda_rigid_2000.pdf?sequence=1
    # get estimates for roll and pitch
    # calculate rate of change of r, p y
    control_input = R @ np.array([gyro.x, gyro.y, gyro.z]).reshape(-1, 1)
    return control_input


def main():
    waypoints, time, sensors = generate_environment(
        f"{ABS_PATH}/../../data/5e15730aa280850006f3d005.txt"
    )
    start_pos = waypoints[time[0]]

    accel_sensor = sensors["accelerometer"]
    gyro_sensor = sensors["gyroscope"]

    # state = [x, y, z, vx, vy, vz, ax, ay, az, r, p, y, rb, pb, yb]
    # observation = [ax, ay, ax]
    process_noise_covariance = np.diag(
        np.hstack(
            [
                np.random.normal(size=(6,)),
                np.diag(accel_sensor.covariance),
                np.diag(gyro_sensor.covariance),
                np.random.normal(size=(3,)),
            ]
        )
    )
    matrix_transform = np.array(
        [
            # x, y, z, vx, vy, vz, ax, ay, az, r, p, y, r_bias, p_bias, y_bias
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    kf = KalmanFilter(
        matrix_transform=matrix_transform,
        process_noise_covariance=process_noise_covariance,
    )

    first_iter = True
    for i, t in enumerate(time):
        if (accel_t := accel_sensor.poll(t)) is None or (
            gyro_t := gyro_sensor.poll(t)
        ) is None:
            continue

        if first_iter:
            kf.init_state(
                t=t,
                state=np.array(
                    [start_pos.x, start_pos.y, start_pos.z]
                    + [1e-5] * 3
                    + [accel_t.x, accel_t.y, accel_t.z]
                    + [1e-5] * 6
                ).reshape(-1, 1),
                estimate_uncertainty=np.full(shape=(15, 15), fill_value=0.99),
            )
            first_iter = False
            continue

        dt = (t - time[i - 1]).total_seconds()

        control_input = generate_control_input(t, accel_t, gyro_t)
        control_transform = np.array(
            [
                # r_dot      p_dot      y_dot
                [0, 0, 0],  # x
                [0, 0, 0],  # y
                [0, 0, 0],  # z
                [0, 0, 0],  # vx
                [0, 0, 0],  # vy
                [0, 0, 0],  # vz
                [0, 0, 0],  # ax
                [0, 0, 0],  # ay
                [0, 0, 0],  # az
                [dt, 0, 0],  # r
                [0, dt, 0],  # p
                [0, 0, dt],  # y
                [0, 0, 0],  # rb
                [0, 0, 0],  # pb
                [0, 0, 0],  # yb
            ]
        )

        state_transition = np.array(
            [
                # x  y  z   vx  vy  vz  ax  ay  az  r   p   y rb pb yb
                [1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                [0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0, 0, 0, 0, 0, 0],  # y
                [0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0, 0, 0, 0, 0],  # z
                [0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],  # vx
                [0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],  # vy
                [0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],  # vz
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # ax
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # ay
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # az
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -dt, 0, 0],  # r
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -dt, 0],  # p
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -dt],  # y
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # rb
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # pb
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # yb
            ]
        )
        observation = np.array([accel_t.x, accel_t.y, accel_t.z]).reshape(-1, 1)
        measurement_uncertainty = np.diag(
            [
                accel_sensor.maximum_range,
                accel_sensor.maximum_range,
                accel_sensor.maximum_range,
            ],
        )

        kf.run(
            t=t,
            control_input=control_input,
            control_transform=control_transform,
            measurement_uncertainty=measurement_uncertainty,
            state_transition_transform=state_transition,
            observation=observation,
        )

    state = kf.state_history
    uncertainty = kf.uncertainty_history
    gain = kf.gain_history
    plot_3d_timeseries(time, state[:, 0], state[:, 1], state[:, 2])
    plot_3d_timeseries(time, state[:, 9], state[:, 10], state[:, 11])
    plot_3d_timeseries(time, gain[:, 6, 0], gain[:, 7, 1], gain[:, 8, 2])
    plot_3d_timeseries(time, uncertainty[:, 0], uncertainty[:, 1], uncertainty[:, 2])


main()
