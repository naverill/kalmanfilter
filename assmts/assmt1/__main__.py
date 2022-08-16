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


class Measurement:
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        accuracy: float,
    ):
        self.x = float(x)
        self.y = float(y)
        self.x = float(x)


class MeasurementUncalibrated(Measurement):
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        x_corrected: float,
        y_corrected: float,
        z_corrected: float,
        accuracy: float,
    ):
        self.x_corrected = float(x_corrected)
        self.y_corrected = float(y_corrected)
        self.z_corrected = float(z_corrected)
        super().__init__(x, y, z, accuracy)


class Sensor:
    def __init__(
        self,
        id: str,
        type: str,
        version: str,
        vendor: str,
        resolution: float,
        power: float,
        maximumRange: float,
    ):
        self.id = id
        self.type = type
        self.version = version
        self.vendor = vendor
        self.resolution = resolution
        self.power = power
        self.maximum_range = maximumRange

    _MEASUREMENTS = {}

    def poll(self):
        return self._MEASUREMENTS.remove(0)


class Waypoint(Sensor):
    def add_reading(
        self,
        timestamp: int,
        x: float,
        y: float,
    ):
        self._MEASUREMENTS[timestamp] = {"x": float(x), "y": float(y)}


class Accelerometer(Sensor):
    def add_reading(
        self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        accuracy: float,
    ):
        """
        x: Acceleration force along the x axis (including gravity) (m/s^2)
        y: Acceleration force along the y axis (including gravity) (m/s^2)
        z: Acceleration force along the z axis (including gravity) (m/s^2)
        """
        self._MEASUREMENTS[timestamp] = Measurement(**{
            "x": x,
            "y": y,
            "z": z,
            "accuracy": float(accuracy),
        })


class AccelerometerUncalibrated(Accelerometer):
    def add_reading(
        self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        x_corrected: float,
        y_corrected: float,
        z_corrected: float,
        accuracy: float,
    ):
        """
        x: acceleration along the X axis without any bias compensation (m/s^2)
        y: acceleration along the Y axis without any bias compensation (m/s^2)
        z: acceleration along the Z axis without any bias compensation (m/s^2)
        x_corrected: acceleration along the X axis with estimated bias compensation  (m/s^2)
        y_corrected: acceleration along the Y axis with estimated bias compensation  (m/s^2)
        z_corrected: acceleration along the Z axis with estimated bias compensation  (m/s^2)


        """
        self._MEASUREMENTS[timestamp] = MeasurementUncalibrated(**{
            "x": x,
            "y": y,
            "z": z,
            "x_corrected": x_corrected,
            "y_corrected": y_corrected,
            "z_corrected": z_corrected,
            "accuracy": accuracy
        })


class Gyroscope(Sensor):
    def add_reading(
        self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        accuracy: float,
    ):
        """
        x: Rate of rotation around the x axis (rad/s)
        y: Rate of rotation around the y axis (rad/s)
        z: Rate of rotation around the z axis (rad/s)
        """
        self._MEASUREMENTS[timestamp] = Measurement(**{
            "x": x,
            "y": y,
            "z": z,
            "accuracy": accuracy,
        })


class GyroscopeUncalibrated(Sensor):
    def add_reading(
        self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        x_corrected: float,
        y_corrected: float,
        z_corrected: float,
        accuracy: float,
    ):
        """
        x: Rate of rotation (without drift compensation) around the x axis (rad/s)
        y: Rate of rotation (without drift compensation) around the y axis (rad/s)
        z: Rate of rotation (without drift compensation) around the z axis (rad/s)
        x_corrected: Estimated drift around the x axis (rad/s)
        y_corrected: Estimated drift around the y axis (rad/s)
        z_corrected: Estimated drift around the z axis (rad/s)
        """
        self._MEASUREMENTS[timestamp] = MeasurementUncalibrated(**{
            "x": x,
            "y": y,
            "z": z,
            "x_corrected": x_corrected,
            "y_corrected": y_corrected,
            "z_corrected": z_corrected,
            "accuracy": accuracy,
        })


class Magnetometer(Sensor):
    def add_reading(
        self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        accuracy: float,
    ):
        """
        x: Geomagnetic field strength along the x axis (μT)
        y: Geomagnetic field strength along the y axis (μT)
        z: Geomagnetic field strength along the z axis (μT)
        """
        self._MEASUREMENTS[timestamp] = Measurement(**{
            "x": x,
            "y": y,
            "z": z,
            "accuracy": accuracy,
        })


class MagnetometerUncalibrated(Sensor):
    def add_reading(
        self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        x_corrected: float,
        y_corrected: float,
        z_corrected: float,
        accuracy: float,
    ):
        """
        x: Geomagnetic field strength (without hard iron calibration) along the x axis (μT)
        y: Geomagnetic field strength (without hard iron calibration) along the y axis (μT)
        z: Geomagnetic field strength (without hard iron calibration) along the z axis (μT)
        x_corrected: Iron bias estimation along the x axis (μT)
        y_corrected: Iron bias estimation along the x axis (μT)
        z_corrected: Iron bias estimation along the x axis (μT)
        """
        self._MEASUREMENTS[timestamp] = MeasurementUncalibrated(**{
            "x": x,
            "y": y,
            "z": z,
            "x_corrected": x_corrected,
            "y_corrected": y_corrected,
            "z_corrected": z_corrected,
            "accuracy": accuracy,
        })


class RotationVector(Sensor):
    def add_reading(
        self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        accuracy: float,
    ):
        """
        x: Rotation vector component along the x axis (x * sin(θ/2))
        y: Rotation vector component along the y axis (y * sin(θ/2))
        z: Rotation vector component along the z axis (z * sin(θ/2)).
        """
        self._MEASUREMENTS[timestamp] = Measurement(**{
            "x": x,
            "y": y,
            "z": z,
            "accuracy": accuracy,
        })


def create_sensor(config):
    params = {}
    config = re.findall("([a-zA-Z]*):([a-zA-Z0-9 .]*)", config)
    for field, value in config:
        params[field] = value

    name = params.pop("name")
    id, type = name[:name.find(" ")], name[name.find(" ") + 1:]
    params.update({"id": id, "type": type})
    sensor = eval(type.replace(" ", ""))
    return sensor(**params)


def create_measurement(sensors: dict, reading: str):
    data = reading.rstrip("\n").split("\t")
    t = datetime.fromtimestamp(int(data[0]) / 1000.0)
    reading_type = data[1].lstrip("TYPE_")
    measurement = data[2:]

    sensor_type = reading_type.replace("MAGNETIC_FIELD", "MAGNETOMETER")\
        .lower()\
        .replace("_", " ")

    if sensor := sensors.get(sensor_type):
        sensor.add_reading(t, *measurement)
    return t


def generate_environment(file_path):
    sensors = {
        "waypoint": Waypoint("W1", "Waypoint", "1", "BOSCH", 0., 1, 0),
        "rotation vector": RotationVector("R1", "RotationVector", "1", "BOSCH", 0., 1, 0),
    }
    timestamps = set()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            if line.find("type:") != -1:
                sensor = create_sensor(line)
                if sensor:
                    sensors[sensor.type.lower()] = sensor
            else:
                continue
        else:
            t = create_measurement(sensors, line)
            timestamps.add(t)
    return sorted(list(timestamps)), sensors


ABS_PATH = Path(__file__).parent.resolve()
timestamps, sensors = generate_environment(f"{ABS_PATH}/../../data/5e15730aa280850006f3d005.txt")
print(timestamps)

