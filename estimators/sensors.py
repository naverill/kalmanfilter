from datetime import datetime
from math import inf

import numpy as np


class Measurement:
    def __init__(
        self,
        timestamp: datetime,
        x: float,
        y: float,
        z: float,
        accuracy: int = None,
    ):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        self._accuracy = int(accuracy) if accuracy else None

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    @property
    def accuracy(self) -> int:
        """
        Indicates that the sensor is reporting data with high (3), medium (2), or low (1) accuracy
        """
        return self._accuracy

    @property
    def matrix(self) -> np.array:
        return np.array([self.x, self.y, self.z]).reshape((-1, 1))


class MeasurementUncalibrated(Measurement):
    def __init__(
        self,
        timestamp: datetime,
        x: float,
        y: float,
        z: float,
        x_corrected: float = None,
        y_corrected: float = None,
        z_corrected: float = None,
        accuracy: float = None,
    ):
        self._x_corrected = float(x_corrected) if x_corrected is not None else None
        self._y_corrected = float(y_corrected) if y_corrected is not None else None
        self._z_corrected = float(z_corrected) if z_corrected is not None else None
        self._matrix_corrected = None
        self._matrix_full = None
        super().__init__(timestamp, x, y, z, accuracy)

    @property
    def x_corrected(self) -> float:
        return self._x_corrected

    @property
    def y_corrected(self) -> float:
        return self._y_corrected

    @property
    def z_corrected(self) -> float:
        return self._z_corrected

    @property
    def matrix_corrected(self) -> np.array:
        if self._matrix_corrected is None:
            self._matrix_corrected = np.array(
                [self.x_corrected, self.y_corrected, self.z_corrected]
            ).reshape(-1, 1)
        return self._matrix_corrected

    @property
    def matrix_full(self) -> np.array:
        if self._matrix_full is None:
            self._matrix_full = np.vstack([self.matrix, self.matrix_corrected])
        return self._matrix_full


class Waypoint(Measurement):
    def __init__(self, timestamp: datetime, x: float, y: float, z: float = 0):
        super().__init__(timestamp, x, y, z, 3)


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
        self.resolution = float(resolution)
        self.power = float(power)
        self.maximum_range = float(maximumRange)
        self._covariance = None
        self._measurements: dict[datetime, Measurement] = {}

    def add_reading(
        self,
        timestamp: datetime,
        x: float,
        y: float,
        z: float,
        accuracy: float = None,
    ):
        self._measurements[timestamp] = Measurement(
            timestamp,
            x,
            y,
            z,
            accuracy,
        )

    def poll(self, timestamp: datetime) -> Measurement:
        return self._measurements.get(timestamp)

    @property
    def covariance(self):
        if self._covariance is None:
            self._covariance = np.cov(
                np.hstack([m.matrix for m in self._measurements.values()])
            )
        return self._covariance


class SensorUncalibrated(Sensor):
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
        super().__init__(id, type, version, vendor, resolution, power, maximumRange)
        self._covariance_corrected = None
        self._covariance_full = None

    def add_reading(
        self,
        timestamp: datetime,
        x: float,
        y: float,
        z: float,
        x_corrected: float = None,
        y_corrected: float = None,
        z_corrected: float = None,
        accuracy: float = None,
    ):
        self._measurements[timestamp] = MeasurementUncalibrated(
            timestamp,
            x,
            y,
            z,
            x_corrected,
            y_corrected,
            z_corrected,
            accuracy,
        )

    @property
    def covariance_corrected(self):
        if self._covariance_corrected is None:
            self._covariance_corrected = np.cov(
                np.hstack([m.matrix_corrected for m in self._measurements.values()])
            )
        return self._covariance_corrected

    @property
    def covariance_full(self):
        if self._covariance_full is None:
            self._covariance_full = np.cov(
                np.hstack([m.matrix_full for m in self._measurements.values()])
            )
        return self._covariance_full


class GPS(Sensor):
    """
    x: GPS position along the x axis
    y: GPS position along the y axis
    z: GPS position along the z axis
    """

    pass


class Accelerometer(Sensor):
    """
    x: Acceleration force along the x axis (including gravity) (m/s^2)
    y: Acceleration force along the y axis (including gravity) (m/s^2)
    z: Acceleration force along the z axis (including gravity) (m/s^2)
    """

    pass


class AccelerometerUncalibrated(Accelerometer, SensorUncalibrated):
    """
    x: acceleration along the X axis without any bias compensation (m/s^2)
    y: acceleration along the Y axis without any bias compensation (m/s^2)
    z: acceleration along the Z axis without any bias compensation (m/s^2)
    x_corrected: acceleration along the X axis with estimated bias
        compensation  (m/s^2)
    y_corrected: acceleration along the Y axis with estimated bias
        compensation  (m/s^2)
    z_corrected: acceleration along the Z axis with estimated bias
        compensation  (m/s^2)
    """

    pass


class Gyroscope(Sensor):
    """
    x: Rate of rotation around the x axis (rad/s)
    y: Rate of rotation around the y axis (rad/s)
    z: Rate of rotation around the z axis (rad/s)
    """

    pass


class GyroscopeUncalibrated(Gyroscope, SensorUncalibrated):
    """
    x: Rate of rotation (without drift compensation)
        around the x axis (rad/s)
    y: Rate of rotation (without drift compensation)
        around the y axis (rad/s)
    z: Rate of rotation (without drift compensation)
        around the z axis (rad/s)
    x_corrected: Estimated drift around the x axis (rad/s)
    y_corrected: Estimated drift around the y axis (rad/s)
    z_corrected: Estimated drift around the z axis (rad/s)
    """

    pass


class Magnetometer(Sensor):
    """
    x: Geomagnetic field strength along the x axis (??T)
    y: Geomagnetic field strength along the y axis (??T)
    z: Geomagnetic field strength along the z axis (??T)
    """

    pass


class MagnetometerUncalibrated(Magnetometer, SensorUncalibrated):
    """
    x: Geomagnetic field strength (without hard iron calibration)
        along the x axis (??T)
    y: Geomagnetic field strength (without hard iron calibration)
        along the y axis (??T)
    z: Geomagnetic field strength (without hard iron calibration)
        along the z axis (??T)
    x_corrected: Iron bias estimation along the x axis (??T)
    y_corrected: Iron bias estimation along the x axis (??T)
    z_corrected: Iron bias estimation along the x axis (??T)
    """

    pass


class RotationVector(Sensor):
    """
    x: Rotation vector component along the x axis (x * sin(??/2))
    y: Rotation vector component along the y axis (y * sin(??/2))
    z: Rotation vector component along the z axis (z * sin(??/2)).
    """

    pass
