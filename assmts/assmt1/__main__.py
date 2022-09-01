"""
TODO:
    - extend to EKF
"""
import os
import random
from datetime import datetime
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
from estimators.utils import (
    filter_acceleration,
    generate_body_estimates,
    generate_euler_rates,
)
from estimators.visualise import (
    plot_2d_timeseries,
    plot_3d_timeseries,
    plot_uncertainty_timeseries,
)

# random.seed(3333)
ABS_PATH = Path(__file__).parent.resolve()
G = 9.81


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

        lin_accel_t, gravity = filter_acceleration(t, gravity, accel_t)

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
        pose_est = generate_body_estimates(t, accel_t, mag_t)
        rot_rate_est = generate_euler_rates(t, pose_est, gyro_t)
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

        observation = np.array(
            [
                lin_accel_t.x,
                lin_accel_t.y,
                lin_accel_t.z,
                pose_est.x,
                pose_est.y,
                pose_est.z,
                rot_rate_est.x,
                rot_rate_est.y,
                rot_rate_est.z,
            ]
        ).reshape(-1, 1)
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
