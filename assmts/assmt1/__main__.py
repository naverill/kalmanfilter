"""
TODO:
    - extend to EKF
"""
import os
import random
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from utils import generate_environment, generate_sensors

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
from estimators.visualise import plot_2d_timeseries, plot_3d_timeseries

ABS_PATH = Path(__file__).parent.resolve()
random.seed(12345)
G = 9.81


def generate_control_input(
    time: datetime,
    r_est: float,
    p_est: float,
    gyro: Measurement,
) -> np.array:
    """
    control vector is  euler angle velocities of the IMU in world frame

    pitch, roll estimation:
    https://escholarship.org/content/qt5rs5t0sf/qt5rs5t0sf_noSplash_e1588dedf177d86a3652374bc997314f.pdf

    body rate to euler transformation:
    https://au.mathworks.com/help/aeroblks/customvariablemass6dofeulerangles.html
    """
    # calculate rate of change of euler angles
    R = np.array(
        [
            [1, np.sin(r_est) * np.tan(p_est), np.cos(r_est) * np.tan(p_est)],
            [0, np.cos(p_est), -np.sin(p_est)],
            [0, np.sin(p_est) / np.cos(r_est), np.cos(p_est) / np.cos(r_est)],
        ]
    )
    control_input = R @ np.array([gyro.x, gyro.y, gyro.z]).reshape(-1, 1)
    return control_input


def filter_accelation(
    timestep: datetime, gravity: Measurement, accel: Measurement, alpha: float = 0.8
) -> [Measurement, Measurement]:
    """
    Filter to remove gravitational acceleration from the accelerometer reasings

    Involves:
        - Isolating the force of gravity with a low-pass filter
        - Removing the gravity contribution with a high-pass filter
    """
    gravity = Measurement(
        timestep,
        x=alpha * gravity.x + (1 - alpha) * accel.x,
        y=alpha * gravity.y + (1 - alpha) * accel.y,
        z=alpha * gravity.z + (1 - alpha) * accel.z,
    )
    lin_accel_t = Measurement(
        timestep,
        x=accel.x - gravity.x,
        y=accel.y - gravity.y,
        z=accel.z - gravity.z,
    )
    return lin_accel_t, gravity


def main():
    fdir = f"{ABS_PATH}/../../data/indoor_robot/"
    config_file = "sensor_config.txt"
    files = [f for f in os.listdir(fdir) if f != config_file]

    sensors = generate_sensors(fdir + config_file)
    waypoints, time = generate_environment(
        file_path=fdir + random.choice(files), sensors=sensors
    )
    start_pos = (
        waypoints[time[0]] if waypoints.get(time[0]) else Waypoint(time[0], 0, 0)
    )

    accel_sensor = sensors["accelerometer"]
    gyro_sensor = sensors["gyroscope"]

    # state = [x, y, z, vx, vy, vz, ax, ay, az, r, p, y, rb, pb, yb]
    # observation = [ax, ay, ax]
    process_noise_covariance = np.diag(
        np.hstack(
            [
                np.array(
                    [
                        accel_sensor.maximum_range,
                    ]
                    * 6
                ),
                np.diag(accel_sensor.covariance),
                np.diag(gyro_sensor.covariance),
                np.array(
                    [
                        gyro_sensor.maximum_range,
                    ]
                    * 3
                ),
            ]
        )
    )
    matrix_transform = np.array(
        [
            # x, y, z, vx, vy, vz, ax, ay, az, r, p, y, r_bias, p_bias, y_bias
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # ax
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # ay
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # az
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # az
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # az
        ]
    )
    kf = KalmanFilter(
        matrix_transform=matrix_transform,
        process_noise_covariance=process_noise_covariance,
    )

    first_iter = True
    gravity = Measurement(time[0], 0, 0, 0)
    prev_t = None
    for t in time:
        if (accel_t := accel_sensor.poll(t)) is None or (
            gyro_t := gyro_sensor.poll(t)
        ) is None:
            continue

        lin_accel_t, gravity = filter_accelation(t, gravity, accel_t)

        if first_iter:
            kf.init_state(
                t=t,
                state=np.array(
                    [start_pos.x, start_pos.y, start_pos.z]
                    + [1e-5] * 3
                    + [lin_accel_t.x, lin_accel_t.y, lin_accel_t.z]
                    + [1e-5] * 6
                ).reshape(-1, 1),
                estimate_uncertainty=np.diag(
                    [
                        1000,
                    ]
                    * 15
                ),
            )
            first_iter = False
            prev_t = t
            continue

        dt = (t - prev_t).total_seconds()

        # get estimates for roll and pitch
        r_est = np.arctan2(accel_t.y, np.sqrt(accel_t.x**2 + accel_t.z**2))
        p_est = np.arctan2(accel_t.x, np.sqrt(accel_t.y**2 + accel_t.z**2))
        control_input = generate_control_input(t, r_est, p_est, gyro_t)
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
        observation = np.array(
            [lin_accel_t.x, lin_accel_t.y, lin_accel_t.z, r_est, p_est]
        ).reshape(-1, 1)
        observation_uncertainty = np.diag(
            [
                accel_sensor.maximum_range,
                accel_sensor.maximum_range,
                accel_sensor.maximum_range,
                accel_sensor.maximum_range,
                accel_sensor.maximum_range,
            ],
        )

        kf.run(
            t=t,
            control_input=control_input,
            control_transform=control_transform,
            measurement_uncertainty=observation_uncertainty,
            state_transition_transform=state_transition,
            observation=observation,
        )
        prev_t = t

    plot_state(kf)


def plot_state(kf: KalmanFilter):
    time: list[datetime] = kf.timesteps
    state: NDArray = kf.state_history
    uncertainty: NDArray = kf.uncertainty_history
    gain: NDArray = kf.gain_history
    inputs: NDArray = kf.input_history
    observations: NDArray = kf.observation_history

    plot_3d_timeseries(
        time,
        x=state[:, 0],
        y=state[:, 1],
        z=state[:, 2],
        title="Position over time",
    )
    plot_3d_timeseries(
        time, gain[:, 6, 0], gain[:, 7, 1], gain[:, 8, 2], "Kalman Gain over time"
    )
    return
    plot_2d_timeseries(
        time,
        x=inputs[:, 0],
        y=inputs[:, 1],
        z=inputs[:, 2],
        title="Euler rates over time",
    )
    plot_2d_timeseries(
        time,
        x=observations[:, 0],
        y=observations[:, 1],
        z=observations[:, 2],
        title="Linear Acceleration over time",
    )
    plot_2d_timeseries(
        time,
        x=state[:, 6],
        y=state[:, 7],
        z=state[:, 8],
        title="Filtered Acceleration over time",
    )
    plot_3d_timeseries(
        time,
        uncertainty[:, 0],
        uncertainty[:, 1],
        uncertainty[:, 2],
        "Position uncertainty over time",
    )


main()
