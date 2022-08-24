## Data Summary
The dataset for this competition consists of dense indoor signatures of WiFi, geomagnetic field, iBeacons etc., as well as ground truth (waypoint) (locations) collected from hundreds of buildings in Chinese cities. The data found in path trace files (*.txt) corresponds to an indoor path between position p_1 and p_2 walked by a site-surveyor.

During the walk, an Android smartphone is held flat in front of the surveyors body, and a sensor data recording app is running on the device to collect IMU (accelerometer, gyroscope) and geomagnetic field (magnetometer) readings, as well as WiFi and Bluetooth iBeacon scanning results. A detailed description of the format of trace file is shown, along with other details and processing scripts, at this github link. In addition to raw trace files, floor plan metadata (e.g., raster image, size, GeoJSON) are also included for each floor.

A note on data quality: In the training files, you may find occasionally that a line is missing the ending newline character, causing it to run on to the next line. It is up to you how you want to handle this issue. This issue is not found in the test data.

[data source](https://www.kaggle.com/competitions/indoor-location-navigation/data)
