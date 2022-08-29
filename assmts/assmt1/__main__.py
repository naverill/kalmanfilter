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
from estimators.visualise import (
    plot_2d_timeseries,
    plot_3d_timeseries,
    plot_uncertainty_timeseries,
)

# random.seed(3333)
ABS_PATH = Path(__file__).parent.resolve()
G = 9.81


def generate_body_estimates(accel: Measurement, mag: Measurement):
    r_est = np.arctan2(accel.y, np.sqrt(accel.x**2 + accel.z**2))
    p_est = np.arctan2(accel.x, np.sqrt(accel.y**2 + accel.z**2))
    body_rot_x = np.array(
        [np.cos(p_est), np.sin(r_est) * np.sin(p_est), np.cos(r_est) * np.sin(p_est)]
    )
    mag_x = body_rot_x @ np.array([mag.x, mag.y, mag.z])
    body_rot_y = np.array([0, np.cos(r_est), np.sin(r_est)])
    mag_y = body_rot_y @ np.array([mag.x, mag.y, mag.z])
    y_est = np.arctan2(-mag_y, mag_x)
    return r_est, p_est, y_est


def generate_euler_rates(
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
    euler_rate = R @ np.array([gyro.x, gyro.y, gyro.z]).reshape(-1, 1)
    return euler_rate


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
    fdir = f"{ABS_PATH}/../../data/indoor_robot/train/"
    config_file = "sensor_config.txt"
    files = [f for f in os.listdir(fdir) if f != config_file]
    fpath = fdir + random.choice(files)

    sensors = generate_sensors(fdir + config_file)
    waypoints, time = generate_environment(file_path=fpath, sensors=sensors)
    start_pos = (
        waypoints[time[0]] if waypoints.get(time[0]) else Waypoint(time[0], 0, 0)
    )

    accel_sensor = sensors["accelerometer"]
    gyro_sensor = sensors["gyroscope"]
    mag_sensor = sensors["magnetometer"]

    matrix_transform = np.array(
        [
            # x, y, z, vx, vy, vz, ax, ay, az, r, p, y, vr, vp, vy, r_bias, p_bias, y_bias
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ax
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ay
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # az
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # r
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # p
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # vr
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # vp
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # vy
        ]
    )
    kf = KalmanFilter(
        matrix_transform=matrix_transform,
    )

    first_iter = True
    gravity = Measurement(time[0], 0, 0, G)
    prev_t = None
    for t in time:
        if (
            (accel_t := accel_sensor.poll(t)) is None
            or (gyro_t := gyro_sensor.poll(t)) is None
            or (mag_t := mag_sensor.poll(t)) is None
        ):
            continue

        lin_accel_t, gravity = filter_accelation(t, gravity, accel_t)

        if first_iter:
            kf.init_state(
                t=t,
                state=np.array(
                    [start_pos.x, start_pos.y, start_pos.z]
                    + [1e-5] * 3
                    + [lin_accel_t.x, lin_accel_t.y, lin_accel_t.z]
                    + [1e-5] * 9
                ).reshape(-1, 1),
                estimate_uncertainty=np.diag(
                    [
                        500,
                    ]
                    * 18
                ),
            )
            first_iter = False
            prev_t = t
            continue

        dt = (t - prev_t).total_seconds()

        # get estimates for roll and pitch
        r_est, p_est, y_est = generate_body_estimates(accel_t, mag_t)
        euler_rates = generate_euler_rates(t, r_est, p_est, gyro_t)
        control_input = np.array([0]).reshape(-1, 1)
        control_transform = np.array([0] * 18).reshape(-1, 1)

        state_transition = np.array(
            [
                # x  y  z   vx  vy  vz  ax  ay  az  r   p   y vr vp vy rb pb yb
                [
                    1,
                    0,
                    0,
                    dt,
                    0,
                    0,
                    0.5 * dt**2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    0,
                    0,
                    dt,
                    0,
                    0,
                    0.5 * dt**2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                    0,
                    dt,
                    0,
                    0,
                    0.5 * dt**2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # vx
                [0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # vy
                [0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # vz
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ax
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ay
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # az
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, -1, 0, 0],  # r
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, -1, 0],  # p
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, -1],  # y
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # vr
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # vp
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # vy
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # rb
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # pb
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # yb
            ]
        )
        process_noise_covariance = (
            state_transition
            @ np.diag(
                np.hstack(
                    [
                        np.diag(accel_sensor.covariance),
                        np.diag(accel_sensor.covariance),
                        np.diag(accel_sensor.covariance),
                        np.diag(gyro_sensor.covariance),
                        np.diag(gyro_sensor.covariance)[:2],
                        np.diag(mag_sensor.covariance)[2:],
                        np.diag(gyro_sensor.covariance)[:2],
                        np.diag(mag_sensor.covariance)[2:],
                    ]
                )
            )
            @ state_transition.T
        )

        observation = np.vstack(
            [
                np.array(
                    [lin_accel_t.x, lin_accel_t.y, lin_accel_t.z, r_est, p_est, y_est]
                ).reshape(-1, 1),
                euler_rates,
            ]
        )
        observation_uncertainty = np.diag(
            [accel_sensor.maximum_range] * 3 + [gyro_sensor.maximum_range] * 6
        )

        kf.run(
            t=t,
            control_input=control_input,
            control_transform=control_transform,
            measurement_uncertainty=observation_uncertainty,
            state_transition_transform=state_transition,
            observation=observation,
            process_noise_covariance=process_noise_covariance,
        )
        prev_t = t

    plot_state(kf, waypoints)


def plot_state(
    kf: KalmanFilter,
    waypoints: dict[datetime, Measurement] = {},
):
    time: list[datetime] = kf.timesteps
    state: NDArray = kf.state_history
    uncertainty: NDArray = kf.uncertainty_history
    gain: NDArray = kf.gain_history
    observations: NDArray = kf.observation_history

    plot_3d_timeseries(
        time,
        x=state[:, 0],
        y=state[:, 1],
        z=state[:, 2],
        x_true=[waypoint.x for waypoint in waypoints.values()],
        y_true=[waypoint.y for waypoint in waypoints.values()],
        z_true=[waypoint.z for waypoint in waypoints.values()],
        title="Position over time",
        scene=dict(zaxis=dict(range=[-10, 10])),
    )
    fields = [
        "X",
        "Y",
        "Z",
        "VX",
        "VY",
        "VZ",
        "AX",
        "AY",
        "AZ",
        "Roll",
        "Pitch",
        "Yaw",
        "Roll Rate",
        "Pitch Rate",
        "Yaw Rate",
        "Roll Bias",
        "Pitch Bias",
        "Yaw Bias",
    ]
    for i, val in enumerate(fields):
        plot_uncertainty_timeseries(
            time,
            state=state[:, i],
            uncertainty=uncertainty[:, i],
            title=f"{val} Uncertainty over time",
        )

    plot_2d_timeseries(
        time,
        x=state[:, 9],
        y=state[:, 10],
        z=state[:, 11],
        x_obs=observations[:, 3],
        y_obs=observations[:, 4],
        z_obs=observations[:, 5],
        title="Euler angles over time",
    )
    plot_2d_timeseries(
        time,
        x=state[:, 12],
        y=state[:, 13],
        z=state[:, 14],
        x_obs=observations[:, 6],
        y_obs=observations[:, 7],
        z_obs=observations[:, 8],
        title="Euler Rates over time",
    )
    plot_3d_timeseries(
        time=time,
        x=gain[:, 6, 0],
        y=gain[:, 7, 1],
        z=gain[:, 8, 2],
        title="Euler angle kalman gain over time",
    )


main()
