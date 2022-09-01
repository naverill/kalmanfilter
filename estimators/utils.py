from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from estimators.sensors import Measurement


def generate_body_estimates(time: datetime, accel: Measurement, mag: Measurement):
    """
    Generate pose estimate

    Returns:
        (roll, pitch, yaw)
    """
    r_est = np.arctan2(accel.y, np.sqrt(accel.x**2 + accel.z**2))
    p_est = np.arctan2(accel.x, np.sqrt(accel.y**2 + accel.z**2))
    body_rot_x = np.array(
        [np.cos(p_est), np.sin(r_est) * np.sin(p_est), np.cos(r_est) * np.sin(p_est)]
    )
    mag_x = body_rot_x @ np.array([mag.x, mag.y, mag.z])
    body_rot_y = np.array([0, np.cos(r_est), np.sin(r_est)])
    mag_y = body_rot_y @ np.array([mag.x, mag.y, mag.z])
    y_est = np.arctan2(-mag_y, mag_x)
    return Measurement(time, r_est, p_est, y_est)


def generate_euler_rates(
    time: datetime,
    pose_est: Measurement,
    gyro: Measurement,
) -> np.array:
    """
    control vector is  euler angle velocities of the IMU in world frame

    body rate to euler rate transformation

    Returns:
        (roll rate, pitch rate, yaw rate)
    """
    # calculate rate of change of euler angles
    R = np.array(
        [
            [
                1,
                np.sin(pose_est.x) * np.tan(pose_est.y),
                np.cos(pose_est.x) * np.tan(pose_est.y),
            ],
            [0, np.cos(pose_est.y), -np.sin(pose_est.y)],
            [
                0,
                np.sin(pose_est.y) / np.cos(pose_est.x),
                np.cos(pose_est.y) / np.cos(pose_est.x),
            ],
        ]
    )
    euler_rate = R @ np.array([gyro.x, gyro.y, gyro.z]).reshape(-1, 1)
    return Measurement(time, euler_rate[0], euler_rate[1], euler_rate[2])


def filter_acceleration(
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
