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


class Sensor:
    def __init__(
        self,
        type: str,
        name: str,
        version: str,
        vendor: str,
        resolution: float,
        power: float,
        maximumRange: float,
    ):
        self.name = name
        self.version = version
        self.vendor = vendor
        self.resolution = resolution
        self.power = power
        self.maximum_range = maximumRange

        sensor_fields = name.split()
        self.id = sensor_fields[0]
        self.sensor_type = sensor_fields[1]
        self.calibrated = True if len(sensor_fields) == 2 else False

    _MEASUREMENTS = []

    def poll():
        pass


class Accelerometer(Sensor):
    def poll(self, reading: str):
        pass


class Gyroscope(Sensor):
    def poll(self, reading: str):
        pass


class Magnetometer(Sensor):
    def add_reading(
        self,
        timestamp: int,
        x_axis: float,
        y_axis: float,
        z_axis: float,
        accuracy: float,
    ):
        self._MEASUREMENTS.append(
            {
                "timestamp": timestamp,
                "x": x_axis,
                "y": y_axis,
                "z": z_axis,
                "accuracy": accuracy,
            }
        )

    def poll(self):
        return self._MEASUREMENTS.remove(0)


def create_sensor(config):
    params = {}
    config = re.search("([a-zA-Z]*):([a-zA-Z0-9 .])*", config)
    for g in config.groups():
        field, value = g.split(":")
        params[field] = value

    sensor = None
    if params["name"].contains("Accelerometer"):
        sensor = Accelerometer
    elif params["name"].contains("Gyroscope"):
        sensor = Gyroscope
    elif params["name"].contains("Magnetometer"):
        sensor = Magnetometer

    return sensor(**params)


def create_measurement(sensors: dict, reading: str):
    data = reading.split("\t")
    t = data[0]
    reading_type = data[1].lstrip("TYPE_")
    measurement = data[2:]

    if reading_type == "MAGNETIC_FIELD":
        sensor_type = "Magnetometer"
        calibrated = False
    elif reading_type == "MAGNETIC_FIELD_UNCALIBRATED":
        sensor_type = "Magnetometer"
        calibrated = True
    elif reading_type == "GYROSCOPE":
        sensor_type = "Gyroscope"
        calibrated = True
    elif reading_type == "GYROSCOPE_UNCALIBRATED":
        sensor_type = "Gyroscope"
        calibrated = False
    elif reading_type == "ACCELEROMETER":
        sensor_type = "Accelerometer"
        calibrated = True
    elif reading_type == "ACCELEROMETER_UNCALIBRATED":
        sensor_type = "Accelerometer"
        calibrated = False

    sensor = filter(
        lambda sensor: sensor.sensor_type == sensor_type and calibrated == calibrated,
        sensors.values(),
    )[0]
    sensor.add_reading(t, *measurement)


def generate_environment(file_path):
    sensors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            if line.contains("type:"):
                sensor = create_sensor(line)
                sensors[sensor.name] = sensor
            else:
                continue
        else:
            create_measurement(sensors, line)


ABS_PATH = Path(__file__).parent.resolve()

generate_environment(f"{ABS_PATH}/../../data/5e15730aa280850006f3d005.txt")


kf = KalmanFilter()
