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
import plotly.graph_objs as go
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
    RotationVector,
    Waypoint,
)

ABS_PATH = Path(__file__).parent.resolve()


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
        np.hstack([np.diag(accel_sensor.covariance), np.diag(gyro_sensor.covariance)])
    )
    matrix_transform = np.array(
        [
            # x, y, z, vx, vy, vz, ax, ay, az, r, p, y
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )
    kf = KalmanFilter(
        matrix_transform=matrix_transform,
        process_noise_covariance=process_noise_covariance,
    )

    init_state = np.array(
        [start_pos.x, start_pos.y, start_pos.z, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ).reshape(-1, 1)

    estimate_uncertainty = np.random.normal(size=init_state.shape)
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

        r = np.arctan2(accel_t.y, np.sqrt(accel_t.x**2 + accel_t.z**2))
        p = np.arctan2(accel_t.x, np.sqrt(np.sqrt(accel_t.y**2 + accel_t.z**2)))
        observation = [accel_t.x, accel_t.y, accel_t.z]
        state_transition = np.array(
            [
                # x  y  z   vx  vy  vz  ax  ay  az  r   p   y
                [1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0, 0, 0, 0],  # x
                [0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0, 0, 0],  # y
                [0, 0, 1, 0, 0, dt, 0, 0, 0, 0.5 * dt**2, 0, 0],  # z
                [0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0],  # vx
                [0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0],  # vy
                [0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0],  # vz
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # ax
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # ay
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # az
                [0, 0, 0, 0, 0, 0, 1, 0, 0, -dt, 0, 0],  # r
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -dt, 0],  # p
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -dt],  # y
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
            ]
        )
        # control vector is  euler angle velocities of the IMU in world frame
        # https://calhoun.nps.edu/bitstream/handle/10945/34427/McGhee_bachmann_zyda_rigid_2000.pdf?sequence=1
        R = np.array(
            [
                [1, np.cos(p) * np.sin(r), np.tan(p) * np.cos(r)],
                [0, np.cos(r), -np.sin(r)],
                [0, np.sin(p) / np.cos(r), np.cos(r) / np.sin(p)],
            ]
        )
        control_input = R @ np.array([gyro_t.x, gyro_t.y, gyro_t.z]).reshape(-1, 1)
        measurement_uncertainty = np.array(
            [
                [accel_sensor.maximum_range, 0, 0, 0, 0, 0],
                [0, accel_sensor.maximum_range, 0, 0, 0, 0],
                [0, 0, accel_sensor.maximum_range, 0, 0, 0],
                [0, 0, 0, gyro_sensor.maximum_range, 0, 0],
                [0, 0, 0, 0, gyro_sensor.maximum_range, 0],
                [0, 0, 0, 0, 0, gyro_sensor.maximum_range],
            ]
        )
        kf.run(
            t=t,
            control_input=control_input,
            control_transform=control_transform,
            measurement_uncertainty=measurement_uncertainty,
            state_transition_transform=state_transition,
            observation=observation,
        )
        print(kf.state())


main()

# EKF implementation

# ekf_estimator = EKF(gyr=sample_data.gyro[:,1:4], acc=sample_data.acce[:,1:4], mag=sample_data.magn[:,1:4], frequency=50.0)
# Initializing the class with the sensor data generates our Quaternion in the Q Variable

# Q = ekf_estimator.Q
# n_samples = sample_data.acce.shape[0]
# new_acce = np.zeros((n_samples, 3))
# Initializing Array to hold the Linear acceleration Vector

# for t in range(n_samples):
#    r = R.from_quat(Q[t])
# Getting a Rotation Matrix from the Quaternions
#    new_acce[t] = np.matmul(r.as_matrix().T,sample_data.acce[t][1:4])
# matmul Rotation Matrix Transpose to orignal Acceleration to produce the clean linear acceleration

# vel_x = integrate.cumtrapz(new_acce[:,0], time_, initial=0)
# vel_y = integrate.cumtrapz(new_acce[:,1], time_, initial=0)
# vel_z = integrate.cumtrapz(new_acce[:,2], time_, initial=0)
#
# pos_x = integrate.cumtrapz(vel_x, time_, initial=0)
# pos_y = integrate.cumtrapz(vel_y, time_, initial=0)
# pos_z = integrate.cumtrapz(vel_z, time_, initial=0)
#
