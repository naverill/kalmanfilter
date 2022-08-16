from datetime import datetime
from math import inf


class Measurement:
    def __init__(
        self,
        timestamp: datetime,
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
        timestamp: datetime,
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
        super().__init__(timestamp, x, y, z, accuracy)


class Waypoint(Measurement):
    def __init__(
        self,
        timestamp: datetime,
        x: float,
        y: float,
    ):
        super().__init__(timestamp, x, y, 0, inf)


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

    _MEASUREMENTS: dict[datetime, Measurement] = {}

    def poll(self, timestamp: datetime) -> Measurement:
        return self._MEASUREMENTS[timestamp]


class Accelerometer(Sensor):
    def add_reading(
        self,
        timestamp: datetime,
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
        self._MEASUREMENTS[timestamp] = Measurement(
            **{
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "z": z,
                "accuracy": float(accuracy),
            }
        )


class AccelerometerUncalibrated(Accelerometer):
    def add_reading(
        self,
        timestamp: datetime,
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
        x_corrected: acceleration along the X axis with estimated bias
            compensation  (m/s^2)
        y_corrected: acceleration along the Y axis with estimated bias
            compensation  (m/s^2)
        z_corrected: acceleration along the Z axis with estimated bias
            compensation  (m/s^2)


        """
        self._MEASUREMENTS[timestamp] = MeasurementUncalibrated(
            **{
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "z": z,
                "x_corrected": x_corrected,
                "y_corrected": y_corrected,
                "z_corrected": z_corrected,
                "accuracy": accuracy,
            }
        )


class Gyroscope(Sensor):
    def add_reading(
        self,
        timestamp: datetime,
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
        self._MEASUREMENTS[timestamp] = Measurement(
            **{
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "z": z,
                "accuracy": accuracy,
            }
        )


class GyroscopeUncalibrated(Sensor):
    def add_reading(
        self,
        timestamp: datetime,
        x: float,
        y: float,
        z: float,
        x_corrected: float,
        y_corrected: float,
        z_corrected: float,
        accuracy: float,
    ):
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
        self._MEASUREMENTS[timestamp] = MeasurementUncalibrated(
            **{
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "z": z,
                "x_corrected": x_corrected,
                "y_corrected": y_corrected,
                "z_corrected": z_corrected,
                "accuracy": accuracy,
            }
        )


class Magnetometer(Sensor):
    def add_reading(
        self,
        timestamp: datetime,
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
        self._MEASUREMENTS[timestamp] = Measurement(
            **{
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "z": z,
                "accuracy": accuracy,
            }
        )


class MagnetometerUncalibrated(Sensor):
    def add_reading(
        self,
        timestamp: datetime,
        x: float,
        y: float,
        z: float,
        x_corrected: float,
        y_corrected: float,
        z_corrected: float,
        accuracy: float,
    ):
        """
        x: Geomagnetic field strength (without hard iron calibration)
            along the x axis (μT)
        y: Geomagnetic field strength (without hard iron calibration)
            along the y axis (μT)
        z: Geomagnetic field strength (without hard iron calibration)
            along the z axis (μT)
        x_corrected: Iron bias estimation along the x axis (μT)
        y_corrected: Iron bias estimation along the x axis (μT)
        z_corrected: Iron bias estimation along the x axis (μT)
        """
        self._MEASUREMENTS[timestamp] = MeasurementUncalibrated(
            **{
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "z": z,
                "x_corrected": x_corrected,
                "y_corrected": y_corrected,
                "z_corrected": z_corrected,
                "accuracy": accuracy,
            }
        )


class RotationVector(Sensor):
    def add_reading(
        self,
        timestamp: datetime,
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
        self._MEASUREMENTS[timestamp] = Measurement(
            **{
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "z": z,
                "accuracy": accuracy,
            }
        )
