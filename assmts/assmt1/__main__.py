"""
TODO:
    - extend to EKF
"""
import re
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import integrate
from scipy.spatial.transform import Rotation as Rot

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


def plot_3d_timeseries(
    time: list[datetime],
    x: list[float],
    y: list[float],
    z: list[float],
    truth_values: list[Measurement],
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
                    colorbar=dict(title="Colorbar"),
                    colorscale="Viridis",
                ),
                mode="markers",
            )
        ]
    )
    fig.show()


def create_sensor(config):
    params = {}
    # Parse config string into parameter list
    config = re.findall("([a-zA-Z]*):([a-zA-Z0-9 .]*)", config)
    for field, value in config:
        params[field] = value

    name = params.pop("name")
    id, type = name[: name.find(" ")], name[name.find(" ") + 1 :]
    params.update({"id": id, "type": type})

    sensor = eval(type.replace(" ", ""))
    return sensor(**params)


def process_measurement(reading: str):
    data = reading.rstrip("\n").split("\t")

    t = datetime.fromtimestamp(int(data[0]) / 1000.0)
    measurement = data[2:]

    reading_type = data[1].lstrip("TYPE_")
    sensor_type = (
        reading_type.replace("MAGNETIC_FIELD", "MAGNETOMETER").lower().replace("_", " ")
    )
    return (
        t,
        sensor_type,
        measurement,
    )


def generate_environment(file_path):
    waypoints = {}
    sensors = {
        "rotation vector": RotationVector(
            "R1", "RotationVector", "1", "BOSCH", 0.0, 1, 0
        ),
    }
    time_ = set()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            if line.find("type:") != -1:
                sensor = create_sensor(line)
                if sensor:
                    sensors[sensor.type.lower()] = sensor
        else:
            t, sensor_type, measurement = process_measurement(line)
            if sensor_type == "waypoint":
                waypoints[t] = Waypoint(t, *measurement)
            elif sensor := sensors.get(sensor_type):
                sensor.add_reading(t, *measurement)
            time_.add(t)
    return waypoints, sorted(list(time_)), sensors


def main():
    waypoints, time, sensors = generate_environment(
        f"{ABS_PATH}/../../data/5e15730aa280850006f3d005.txt"
    )
    start_pos = waypoints[time[0]]

    accel_sensor = sensors["accelerometer"]
    gyro_sensor = sensors["gyroscope"]

    # state = [x, y, z, vx, vy, vz, ax, ay, az, r, p, y, rb, pb, yb]
    # observation = [ax, ay, ax, r, p, y]
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
            # x, y, z, vx, vy, vz, ax, ay, az, r, p, y, rb, pb, yb
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    kf = KalmanFilter(
        matrix_transform=matrix_transform,
        process_noise_covariance=process_noise_covariance,
    )

    init_state = np.array(
        [start_pos.x, start_pos.y, start_pos.z, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ).reshape(-1, 1)

    estimate_uncertainty = np.full(
        shape=(init_state.shape[0], init_state.shape[0]), fill_value=0.99
    )
    kf.init_state(
        t=time[0], state=init_state, estimate_uncertainty=estimate_uncertainty
    )

    for i, t in enumerate(time):
        dt = (t - time[i - 1]).total_seconds()

        accel_t = accel_sensor.poll(t)
        gyro_t = gyro_sensor.poll(t)
        if not accel_t or not gyro_t:
            continue

        # https://escholarship.org/content/qt5rs5t0sf/qt5rs5t0sf_noSplash_e1588dedf177d86a3652374bc997314f.pdf

        # get estimates for roll and pitch
        r_hat = np.arctan2(accel_t.y, np.sqrt(accel_t.x**2 + accel_t.z**2))
        p_hat = np.arctan2(accel_t.x, np.sqrt(np.sqrt(accel_t.y**2 + accel_t.z**2)))
        observation = np.array([accel_t.x, accel_t.y, accel_t.z]).reshape(-1, 1)
        state_transition = np.array(
            [
                # x  y  z   vx  vy  vz  ax  ay  az  r   p   y rb pb yb
                [1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                [0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0, 0, 0, 0, 0, 0],  # y
                [0, 0, 1, 0, 0, dt, 0, 0, 0, 0.5 * dt**2, 0, 0, 0, 0, 0],  # z
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
        control_transform = np.array(
            [
                # dr      dp      dy
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
        # control vector is  euler angle velocities of the IMU in world frame
        # https://calhoun.nps.edu/bitstream/handle/10945/34427/McGhee_bachmann_zyda_rigid_2000.pdf?sequence=1
        R = np.array(
            [
                [1, np.cos(p_hat) * np.sin(r_hat), np.tan(p_hat) * np.cos(r_hat)],
                [0, np.cos(r_hat), -np.sin(r_hat)],
                [0, np.sin(p_hat) / np.cos(r_hat), np.cos(r_hat) / np.sin(p_hat)],
            ]
        )
        # calculate rate of change of r, p y
        control_input = R @ np.array([gyro_t.x, gyro_t.y, gyro_t.z]).reshape(-1, 1)
        measurement_uncertainty = np.diag(
            [
                gyro_sensor.maximum_range,
                gyro_sensor.maximum_range,
                gyro_sensor.maximum_range,
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
    plot_3d_timeseries(time, state[:, 0], state[:, 1], state[:, 2])
    plot_3d_timeseries(time, uncertainty[:, 0], uncertainty[:, 1], uncertainty[:, 2])


main()


# vel_x = integrate.cumtrapz(new_acce[:,0], time_, initial=0)
# vel_y = integrate.cumtrapz(new_acce[:,1], time_, initial=0)
# vel_z = integrate.cumtrapz(new_acce[:,2], time_, initial=0)
#
# pos_x = integrate.cumtrapz(vel_x, time_, initial=0)
# pos_y = integrate.cumtrapz(vel_y, time_, initial=0)
# pos_z = integrate.cumtrapz(vel_z, time_, initial=0)
#
