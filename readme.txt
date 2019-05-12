Satellite Service Volume Simulator
Open Source satellite toolkit

M. Tossaint - 2019

General framework for satellite - ground station - user analysis.
Framework takes care of geometry computations.
Analysis can be added a la carte.

COVERAGE
    0. Ground Track of Satellite
    1. Number of satellites in view over time for user (also spacecraft user)
    2. Number of satellites in view statistics for user grid
    3. Satellite IDs in view over time for selected user (also spacecraft user)
    4. Satellite ground visibility contour at StopDate
    5. Satellite Azimuth and Elevation for selected user (also spacecraft user)
    6. Number of station in view of each satellite, depth of coverage (DOC) based on satellite ground track
    7. Satellite passes time statistics for user grid
    8. Satellite elevation statistics for highest satellite in view for user grid
    9. Output satellite and stations time and position in ECI (for later use)

COMMUNICATION
20. Single ISL link assessment for satellite-satellite or ground-satellite receiver power, bitrate and C/N0
21. Single ISL link assessment for satellite-satellite or ground-satellite bitrate vs. ISL power
22. Multiple ISL geometry between satellites and identifies the worst case communication link

NAVIGATION
0. DOP values for users

EARTH OBSERVATION
0. Swath coverage for satellite(s)

Statistics on plots can be:
- Min
- Mean
- Max
- Std
- Median

![alt text](https://raw.githubusercontent.com/mtossain/satsvs/output/analysis_0.png)