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
    time: datetime, accel: Measurement, gyro: Measurement
) -> np.array:
    """
    control vector is  euler angle velocities of the IMU in world frame
    https://calhoun.nps.edu/bitstream/handle/10945/34427/McGhee_bachmann_zyda_rigid_2000.pdf?sequence=1
    https://escholarship.org/content/qt5rs5t0sf/qt5rs5t0sf_noSplash_e1588dedf177d86a3652374bc997314f.pdf
    https://liqul.github.io/blog/assets/rotation.pdf
    https://au.mathworks.com/help/aeroblks/customvariablemass6dofeulerangles.html
    """
    # get estimates for roll and pitch
    r_est = np.arctan2(accel.y, np.sqrt(accel.x**2 + accel.z**2))
    p_est = np.arctan2(accel.x, np.sqrt(accel.y**2 + accel.z**2))
    R = np.array(
        [
            [1, np.sin(r_est) * np.tan(p_est), np.cos(r_est) * np.tan(p_est)],
            [0, np.cos(p_est), -np.sin(p_est)],
            [0, np.sin(p_est) / np.cos(r_est), np.cos(p_est) / np.cos(r_est)],
        ]
    )
    # calculate rate of change of r, p y
    control_input = R @ np.array([gyro.x, gyro.y, gyro.z]).reshape(-1, 1)
    return control_input


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

    alpha = 0.8
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
    gravity = Measurement(time[0], 0, 0, 0)
    accel = []
    gyro = []
    for i, t in enumerate(time):
        if (accel_t := accel_sensor.poll(t)) is None or (
            gyro_t := gyro_sensor.poll(t)
        ) is None:
            continue

        # Isolate the force of gravity with a low-pass filter
        gravity = Measurement(
            t,
            x=alpha * gravity.x + (1 - alpha) * accel_t.x,
            y=alpha * gravity.y + (1 - alpha) * accel_t.y,
            z=alpha * gravity.z + (1 - alpha) * accel_t.z,
        )
        # Remove the gravity contribution with a high-pass filter
        lin_accel_t = Measurement(
            t,
            x=accel_t.x - gravity.x,
            y=accel_t.y - gravity.y,
            z=accel_t.z - gravity.z,
        )
        accel.append([accel_t.x, accel_t.y, accel_t.z])
        gyro.append([gyro_t.x, gyro_t.y, gyro_t.z])

        if first_iter:
            kf.init_state(
                t=t,
                state=np.array(
                    [start_pos.x, start_pos.y, start_pos.z]
                    + [1e-5] * 3
                    + [lin_accel_t.x, lin_accel_t.y, lin_accel_t.z]
                    + [1e-5] * 6
                ).reshape(-1, 1),
                estimate_uncertainty=np.full(shape=(15, 15), fill_value=0.99),
            )
            first_iter = False
            continue

        dt = (t - time[i - 1]).total_seconds()

        control_input = generate_control_input(t, lin_accel_t, gyro_t)
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
        observation = np.array([lin_accel_t.x, lin_accel_t.y, lin_accel_t.z]).reshape(
            -1, 1
        )
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
    inputs = kf.input_history
    observations = kf.observation_history

    plot_3d_timeseries(
        time,
        x=state[:, 0],
        y=state[:, 1],
        z=state[:, 2],
        x_truth=[point.x for point in waypoints.values()],
        y_truth=[point.y for point in waypoints.values()],
        z_truth=[0 for _ in waypoints.values()],
        title="Position over time",
    )
    plot_3d_timeseries(
        time, gain[:, 6, 0], gain[:, 7, 1], gain[:, 8, 2], "Kalman Gain over time"
    )
    return
    plot_3d_timeseries(
        time,
        uncertainty[:, 0],
        uncertainty[:, 1],
        uncertainty[:, 2],
        "Position uncertainty over time",
    )
    gyro = np.array(gyro)
    accel = np.array(accel)
    plot_2d_timeseries(
        time, x=gyro[:, 0], y=gyro[:, 1], z=gyro[:, 2], title="Gyro over time"
    )
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
        x=accel[:, 0],
        y=accel[:, 1],
        z=accel[:, 2],
        title="Acceleration over time",
    )
    plot_2d_timeseries(
        time,
        x=state[:, 6],
        y=state[:, 7],
        z=state[:, 8],
        title="Filtered Acceleration over time",
    )


main()
