# Satellite Service Volume Simulator
## Open Source satellite, ground station and satellite user tool
### M. Tossaint - 2019 - v1


## Introduction
Framework takes care of geometry computations, satellite propagation, ground station and user rotation in ECI/ECF.

Configuration of the tool can be done in the config.xml file where satellites, ground stations, users, simulation
parameters and analysis are defined. Analysis can be added as wished, the baseline of analysis available are below 
(and explained further below):

### Coverage
- __cov_ground_track__: Ground Track of Satellite on map
- __cov_satellite_pvt__: Output satellite position and velocity in ECI (for later use)
- __cov_satellite_visible__: Number of satellites in view over time for user (also spacecraft user)
- __cov_satellite_visible_grid__: Number of satellites in view statistics for user grid
- __cov_satellite_visible_id__: Satellite IDs in view over time for selected user (also spacecraft user)
- __cov_satellite_contour__: Satellite ground visibility contour at StopDate
- __cov_satellite_sky_angles__: Satellite Azimuth and Elevation for selected user (also spacecraft user)
- __cov_satellite_highest__: Satellite elevation statistics for highest satellite in view for user grid
- __cov_depth_of_coverage__: Number of station in view of each satellite, depth of coverage (DOC) based on satellite ground track
- __cov_pass_time__: Satellite passes time statistics for user grid

To be implemented at a later stage:

### Earth observation
- __obs_swath_coverage__: Swath coverage for satellite(s)
- __obs_time_between_passes__: Time between satellite swath passes

### Communication
- __com_sp2sp_budget__: for satellite-satellite received power, bitrate and C/N0
- __com_sp2gr_budget__: for satellite-groundstation received power, bitrate and C/N0
- __com_sp2sp_worst_link__: Multiple ISL geometry between satellites and identifies the worst case communication link

### Navigation
- __nav_dillution_of_precision__: DOP values for user(s) (also spacecraft user)



## Configuration file
The configuration file is found in the input directory under the name __config.xml__. 
This file contains several main parts:
- The space segment in several constellations
- The ground segment with ground stations servicing the satellites
- The user segment receiving the satellite information at single locations or user grids
- The simulation parameters like start,stop,step time and analysis to be performed.

All units in the configuration file are in SI units, angles are in degrees.

### Space segment
The following xml is used to setup the space segment:
```
<?xml version="1.0" encoding="utf-8"?>
<!--Simulation Scenario-->
<Scenario>
    <SpaceSegment>
    <!--Defines the constellations and satellites that should be simulated-->
        <Constellation>
            <NumOfPlanes>3</NumOfPlanes>
            <NumOfSatellites>30</NumOfSatellites>
            <ConstellationID>1</ConstellationID>
            <ConstellationName>Galileo</ConstellationName>
            <Satellite>
                <SatelliteID>1</SatelliteID>
                <Plane>1</Plane>
                <EpochMJD>54465.5</EpochMJD>
                <SemiMajorAxis>29600318</SemiMajorAxis>
                <Eccentricity>0.00</Eccentricity>
                <Inclination>56.00</Inclination>
                <RAAN>0</RAAN>
                <ArgOfPerigee>0.00</ArgOfPerigee>
                <MeanAnomaly>0</MeanAnomaly>
            </Satellite>
        </Constellation>
    </SpaceSegment>

```
Orbit parameters are either Keplerian or can be defined as a list of satellites in a TLE file (in input directory):
```
<Constellation>
    <NumOfPlanes>3</NumOfPlanes>
    <NumOfSatellites>30</NumOfSatellites>
    <ConstellationID>1</ConstellationID>
    <ConstellationName>OrbComm</ConstellationName>
    <TLEFileName>input/tle.txt</TLEFileName>
</Constellation>
```

### Ground segment
The following xml is used to setup the ground segment:
```
  <GroundSegment>
    <Network>
      <NumStation>2</NumStation>
      <NetworkName>GCC</NetworkName>
      <GroundStation>
        <Type>GCC</Type>
        <ConstellationID>1</ConstellationID>
        <GroundStationID>1</GroundStationID>
        <GroundStationName>OBE</GroundStationName>
        <Latitude>48.0744</Latitude>
        <Longitude>11.262</Longitude>
        <Height>0</Height>
        <ReceiverConstellation>110</ReceiverConstellation>
        <ElevationMask>5</ElevationMask>
      </GroundStation>
    </Network>
  </GroundSegment>
```
Ground stations belong to a constellation <ConstellationID> and can be told to receive from multiple constellations 
<ReceiverConstellation> through a list of True/False separated by commas. The elevation mask is defined as a list of
values seperated by commas, dividing the azimuth circle in equal parts.

### User segment
The following xml is used to setup the user segment:
```
<UserSegment>
    <Type>Grid</Type>
    <LatMin>-90</LatMin>
    <LatMax>90</LatMax>
    <LonMin>-180</LonMin>
    <LonMax>180</LonMax>
    <LatStep>10</LatStep>
    <LonStep>10</LonStep>
    <Height>0</Height>
    <ReceiverConstellation>111</ReceiverConstellation>
    <ElevationMask>5</ElevationMask>
</UserSegment>
```
The user segment can be setup as a grid or single locations:
```
<UserSegment>
    <Type>Static</Type>
    <Latitude>50</Latitude>
    <Longitude>5</Longitude>
    <Height>0</Height>
    <ReceiverConstellation>111</ReceiverConstellation>
    <ElevationMask>5</ElevationMask>
</UserSegment>
```
or as a spacecraft user through a TLE file:
```
<UserSegment>
    <Type>Spacecraft</Type>
    <TLEFileName>C:\Documents and Settings\Michel Tossaint\My Documents\Navigation Data\TLE\TLE_MetopA_2006_12_26.txt</TLEFileName>
    <ElevationMask>20</ElevationMask>
    <ReceiverConstellation>1000</ReceiverConstellation>
</UserSegment>
```
Users can be told to receive from multiple constellations <ReceiverConstellation> through a list of True/False 
separated by commas. The elevation mask is defined as a list of values (one or more) seperated by commas, 
dividing the azimuth circle in equal parts.

### Simulation parameters
The following xml is used to setup the simulation parameters:
```
<SimulationManager>
    <StartDate>2013-05-08 00:00:00</StartDate>
    <StopDate>2013-05-09 00:00:00</StopDate>
    <TimeStep>3600</TimeStep>
    <Analysis>
          <Type>9</Type>
          <ConstellationID>1</ConstellationID>
    </Analysis>
</SimulationManager>
```
The time parameters are in UTC time and TimeStep in seconds. The analysis are described below:

## Analysis parameters

### cov_ground_track
Plots the ground track of one or more satellites over simulation time. The following parameters are needed:
```
<Type>cov_ground_track</Type>
<ConstellationID>1</ConstellationID>
```
Optional is SatelliteID: which satellite to use (ID or NORAD number if a TLE file is used): 
```
<SatelliteID>1</SatelliteID>
```
If this parameter is omitted all the satellites of the constellation are plotted.
<img src="/docs/cov_ground_track.png" alt="cov_ground_track"/>

### cov_satellite_pvt
Plots the first satellite velocity and outputs the position and velocity to the 'output/orbits.txt' file. 
The following parameters are needed:
```
<Type>cov_satellite_pvt</Type>
<ConstellationID>1</ConstellationID>
```
Optional are:
```
<SatelliteID>1</SatelliteID>
``` 
If this parameter is omitted all the satellites of the constellation are output. Additionally an __/output/orbits.txt__ 
file is saved to disk.
<img src="/docs/cov_satellite_pvt.png" alt="cov_satellite_pvt"/>

### cov_satellite_visible
Plots the number of available satellites for the user(s). The following parameters are needed:
```
<Type>cov_satellite_visible</Type>
```
<img src="/docs/cov_satellite_visible.png" alt="cov_satellite_visible"/>

### cov_satellite_visible_grid
Plots the number of available satellites at a user grid. Statistics on plots can be: minimum, mean, maximum, std and
median. The following parameters are needed:
```
<Analysis>
      <Type>cov_satellite_visible_grid</Type>
      <Statistic>Min</Statistic>
</Analysis>

```
<img src="/docs/cov_satellite_visible_grid.png" alt="cov_satellite_visible_grid"/>

### cov_satellite_visible_id
Plots the satellite IDs in view over time for the first user. The following parameters are needed:
```
<Analysis>
    <Type>cov_satellite_visible_id</Type>
    <ConstellationID>1</ConstellationID>
</Analysis>
```
<img src="/docs/cov_satellite_visible_id.png" alt="cov_satellite_visible_id"/>

### cov_satellite_contour
Plots the satellite ground contour on the world map. The following parameters are needed:
```
<Analysis>
    <Type>cov_satellite_contour</Type>
    <ConstellationID>1</ConstellationID>
    <SatelliteID>1</SatelliteID>
    <ElevationMask>20</ElevationMask>
</Analysis>
```
The elevation mask is for the user who has to receive the satellite. The satellite is selected by constellation ID and
satellite ID.
<img src="/docs/cov_satellite_contour.png" alt="cov_satellite_contour"/>

### cov_satellite_sky_angles
Plots the satellite azimuth and elevation over time for the first user. The following parameters are needed:
```
<Analysis>
    <Type>cov_satellite_sky_angles</Type>
    <ConstellationID>3</ConstellationID>
    <SatelliteID>24307</SatelliteID>
</Analysis>
```
The satellite is selected by constellation ID and satellite ID.
<img src="/docs/cov_satellite_sky_angles.png" alt="cov_satellite_sky_angles"/>

### cov_depth_of_coverage
Plots the number of ground stations in view from the satellite over the orbit of the satellites. 
The following parameters are needed:
```
<Analysis>
  <Type>cov_depth_of_coverage</Type>
</Analysis>
```
The elevation mask is taken by the ground station setup.
<img src="/docs/cov_depth_of_coverage.png" alt="cov_depth_of_coverage"/>

### cov_pass_time
Plots the satellite constellation pass time statistics for a user grid. The following parameters are needed:
```
<Analysis>
    <Type>cov_pass_time</Type>
    <ConstellationID>1</ConstellationID>
    <Statistic>Mean</Statistic>
</Analysis>
```
<img src="/docs/cov_pass_time.png" alt="cov_pass_time"/>

### cov_satellite_highest
Plots elevation for the highest satellite in view over a user grid. The following parameters are needed:
```
<Analysis>
    <Type>cov_satellite_highest</Type>
    <ConstellationID>1</ConstellationID>
    <Statistic>Mean</Statistic>
</Analysis>
```
<img src="/docs/cov_satellite_highest.png" alt="cov_satellite_highest"/>









