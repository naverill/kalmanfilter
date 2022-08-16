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
from scipy.spatial.transform import Rotation as R

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
    waypoints = []
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
                waypoints.append(Waypoint(t, *measurement))
            elif sensor := sensors.get(sensor_type):
                sensor.add_reading(t, *measurement)
            time_.add(t)
    return waypoints, sorted(list(time_)), sensors


ABS_PATH = Path(__file__).parent.resolve()
waypoints, time, sensors = generate_environment(
    f"{ABS_PATH}/../../data/5e15730aa280850006f3d005.txt"
)
print(time)
start_pos = waypoints[0]
end_pos = waypoints[-1]

# vel_x = integrate.cumtrapz(new_acce[:,0], time_, initial=0)
# vel_y = integrate.cumtrapz(new_acce[:,1], time_, initial=0)
# vel_z = integrate.cumtrapz(new_acce[:,2], time_, initial=0)
#
# pos_x = integrate.cumtrapz(vel_x, time_, initial=0)
# pos_y = integrate.cumtrapz(vel_y, time_, initial=0)
# pos_z = integrate.cumtrapz(vel_z, time_, initial=0)
#
