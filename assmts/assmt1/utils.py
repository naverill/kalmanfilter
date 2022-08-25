import re
from datetime import datetime

from estimators.sensors import (
    Accelerometer,
    AccelerometerUncalibrated,
    Gyroscope,
    GyroscopeUncalibrated,
    Magnetometer,
    MagnetometerUncalibrated,
    Measurement,
    RotationVector,
    Sensor,
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

    # Grab sensor class from type
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


def generate_sensors(sensor_config: str) -> dict[str, Sensor]:
    sensors = {
        "rotationvector": RotationVector(
            "R1", "RotationVector", "1", "BOSCH", 0.0, 1, 0
        ),
    }

    with open(sensor_config, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.find("type:") != -1:
            sensor = create_sensor(line)
            if sensor:
                sensors[sensor.type.lower()] = sensor

    return sensors


def generate_environment(file_path: str, sensors: dict[str, Sensor]):
    waypoints = {}
    time_ = set()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            continue
        else:
            t, sensor_type, measurement = process_measurement(line)
            if sensor_type == "waypoint":
                waypoints[t] = Waypoint(t, *measurement)
            elif sensor := sensors.get(sensor_type):
                sensor.add_reading(t, *measurement)
            time_.add(t)
    return waypoints, sorted(list(time_))
