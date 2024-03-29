# Satellite Service Volume Simulator
## Open Source satellite, ground station and user tool
### M. Tossaint - 2020 - v2

<img src="/docs/schema.png" alt="schema"/>

## Installation & first run
Download from github, and install the following libraries:
- Numpy, pandas and numba
- Astropy and sgp4
- Basemap and xarray
- Geopandas and shapely
- Itur

To run, edit the config.xml file and run: python main.py

## Introduction
Framework takes care of geometry computations, satellite propagation, ground station and user rotation in ECI/ECF.
It will also automatically compute links between stations and satellites, users and satellites, and between satellites.

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
- __cov_depth_of_coverage__: Depth of station coverage (DOC) based on satellite ground track
- __cov_pass_time__: Satellite passes time statistics for user grid

### Earth observation
- __obs_swath_conical__: Swath coverage for satellite(s) with conical scanners
- __obs_swath_pushbroom__: Swath coverage for satellite(s)
- __obs_sza_subsat__: Solar Zenith Angle (SZA) for all satellite(s) sub satellite point

### Communication
- __com_gr2sp_budget__: For station-satellite received power, losses and C/N0
- __com_sp2sp_budget__: For satellite-satellite received power, losses and C/N0
- __com_doppler__: For satellite-station elevation and doppler

### Navigation
- __nav_dillution_of_precision__: DOP values for user(s) (also spacecraft user)
- __nav_accuracy__: Navigation accuracy (UERE*DOP) values for user(s) (also spacecraft user)

_To be implemented at a later stage:_

Satellite
- __sat_power_subsystem__: Power over orbit including eclipse analysis
- __sat_data_handling__: Data storage and downlink over orbit
- __sat_thermal__: Thermal conditions over orbit
- __sat_attitude_control__: Satellite attitude control over orbit

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
The satellite can also be defined as an Sun Synchronous Satellite (SSO) with:
- LTAN Local Time Ascending Node in hours
- Altitude in m
The inclination is computed based on the SSO assumption. Put the Epoch at midday, ArgOfPerigee at 0 and MeanAnomaly at 0,
so that the satellite passes the equator at ltan requested.
```
<Satellite>
    <SatelliteID>1</SatelliteID>
    <Plane>1</Plane>
    <EpochMJD>58945.15</EpochMJD>
    <Altitude>694000</Altitude>
    <Eccentricity>0.0001402</Eccentricity>
    <LTAN>22.25</LTAN>
    <ArgOfPerigee>0.0</ArgOfPerigee>
    <MeanAnomaly>0.0</MeanAnomaly>
</Satellite>
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
The following xml is used to setup the user segment, here a quadrangle grid:
```
<UserSegment>
    <User>
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
    </User>
</UserSegment>
```
or as single static location:
```
<User>
    <Type>Static</Type>
    <Latitude>50</Latitude>
    <Longitude>5</Longitude>
    <Height>0</Height>
    <ReceiverConstellation>111</ReceiverConstellation>
    <ElevationMask>5</ElevationMask>
</User>
```
or as a polygon surface with manual points as tuples:
```
<User>
    <Type>Polygon</Type>
    <Name>Europe</Name>
    <LatStep>.5</LatStep>
    <LonStep>.5</LonStep>
    <PolygonList>(-24.11, 65.34),(-15.97, 52.45),(-14.89, 42.52),(-15.59, 37.36),(-20.89, 42.52),(-30.06, 43.62),(-36.40, 41.78),(-38.86, 35.15),(-27.21, 31.10),(-20.50, 23.74),(-7.47, 33.68),(6.98, 34.42),(19.68, 31.10),(30.26, 33.68),(39.77, 37.73),(44.35, 40.67),(38.70, 45.09),(37.63, 49.51),(27.04, 57.61),(33.37, 63.87),(39.71, 65.71),(36.88, 72.70),(31.58, 78.22),(13.60, 75.64),(1.28, 68.65),(-9.29, 62.02),(-14.59, 66.81)</PolygonList>
    <Height>0</Height>
    <ReceiverConstellation>1111</ReceiverConstellation>
    <ElevationMask>0</ElevationMask>
</User>
```
or as a polygon surface defined in a shapefile:
```
<User>
    <Type>Polygon</Type>
    <Name>Europe</Name>
    <LatStep>.5</LatStep>
    <LonStep>.5</LonStep>
    <PolygonFile>../input/polygon.shp</PolygonFile>
    <Height>0</Height>
    <ReceiverConstellation>1111</ReceiverConstellation>
    <ElevationMask>0</ElevationMask>
</User>
```
or as a spacecraft user through a TLE file:
```
<UserSegment>
    <Type>Spacecraft</Type>
    <TLEFileName>../input/example_tle_files/TLE_MetopA_2006_12_26.txt</TLEFileName>
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
    <IncludeStation2SpaceLinks>True</IncludeStation2SpaceLinks>
    <IncludeUser2SpaceLinks>True</IncludeUser2SpaceLinks>
    <IncludeSpace2SpaceLinks>False</IncludeSpace2SpaceLinks>
    <OrbitPropagator>SGP4</OrbitPropagator>

    <Analysis>
          <Type>9</Type>
          <ConstellationID>1</ConstellationID>
    </Analysis>
</SimulationManager>
```
The following explanations apply for the parameters:
- The Start/Stop time parameters are in UTC time and TimeStep in seconds. 
- The IncludeStation2SpaceLinks, etc. parameters determine whether links between different objects: sat, station and user are computed. Normally leave these to True so that all analysis works. Time could
be saved by disabling some. 
- The OrbitPropagator determines which propagator: 'Keplerian' or 'SGP4' to take.

The analysis are described below:

## Analysis parameters
In order to run an analysis block it has to be uncommented and the parameters adapted. To add a new analysis to the code the following has to be performed:
- Add a new analysis class at the end of analysis.py
- Use as a template one of the other analysis classes above or the base analysis class definition
- Add the class instantiation at the end of config.py

The specific parameters for the existing analysis are given here below:

### cov_ground_track
Plots the ground track of one or more satellites over simulation time. The following parameters are needed, to plot the ground track of satellites in a constellation:
```
<Type>cov_ground_track</Type>
<ConstellationID>1</ConstellationID>
```
Optional is SatelliteID for selection of one satellite. For a TLE file it is the NORAD number. 
```
<SatelliteID>1</SatelliteID>
```
If this parameter is omitted all the satellites of the constellation are plotted.
<img src="/docs/cov_ground_track.png" alt="cov_ground_track"/>

### cov_satellite_pvt
Plots the first satellite position and velocity and outputs the position and velocity to the 'output/orbits.txt' file. 
The following parameters are needed:
```
<Type>cov_satellite_pvt</Type>
<ConstellationID>1</ConstellationID>
```
Optional is:
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
Plots the satellite(s) ground contour on the world map. The following parameters are needed:
```
<Analysis>
    <Type>cov_satellite_contour</Type>
    <ConstellationID>1</ConstellationID>
    <ElevationMask>20</ElevationMask>
</Analysis>
```
Optional is:
```
    <SatelliteID>1</SatelliteID>
``` 
The elevation mask is for the user who has to receive the satellite. The satellite is selected by constellation ID and
satellite ID or multiple if satellite ID is omitted.
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

### obs_swath_conical
Plots the swath coverage for a conical scanner on one or more satellites defined in the space segment. 
The user segment is used to define the grid of analysis and defines the granularity of the result.
Typically a grid of 1x1 deg is sufficient otherwise for a complete globe the simulation will take lots of time.

The following parameters are needed:
```
<Analysis>
    <Type>obs_swath_conical</Type>
</Analysis>
```
In the constellation part of the space segment are defined the instrument characteristics:
```
<ObsSwathStop>650000.0</ObsSwathStop>
```
as above in meters, at the edge, or in degrees incidence angle:
```
<ObsIncidenceAngleStop>52.0</ObsIncidenceAngleStop>
```
The incidence angle is defined as the angle between the line-of-sight and the nadir vector from the satellite. 
This is not to be confused with the user observation zenith angle.

Optional in the analysis part are:
```
<PolarView>90</PolarView>
<Revisit>True</Revisit>
<Statistic>Mean</Statistic>
<SaveOutput>Numpy</SaveOutput>
```
- PolarView angle: This parameter can be given to see one part of the globe in an stereographic view, eg.  for the polar region.
- Revisit flag: This flag will enable revisit computation after the swath coverage. The statistic will determine
what kind of statistic is displayed per user location.
- SaveOutput: NetCDF or Numpy This flag will enable saving user swath coverage for every timestep.

<img src="/docs/obs_swath_conical.png" alt="obs_swath_conical"/>
<img src="/docs/obs_swath_conical_revisit.png" alt="obs_swath_conical_revisit"/>

### obs_swath_push_broom
Plots the swath coverage for a push broom scanner on one or more satellites defined in the space segment. 
The user segment is used to define the grid of analysis and defines the granularity of the result.
Typically a grid of 1x1 deg is sufficient otherwise for a complete globe the simulation will take lots of time.

The following parameters are needed:
```
<Analysis>
    <Type>obs_swath_push_broom</Type>
</Analysis>
```
In the constellation part of the space segment are defined the instrument characteristics:
```
<ObsSwathStart>250000.0</ObsSwathStart>
<ObsSwathStop>650000.0</ObsSwathStop>
```
as above in meters, at the edge, or in degrees incidence angle:
```
<ObsIncidenceAngleStart>20.0</ObsIncidenceAngleStart>
<ObsIncidenceAngleStop>52.0</ObsIncidenceAngleStop>
```
The incidence angle is defined as the angle between the line-of-sight and the nadir vector from the satellite. 
This is not to be confused with the user observation zenith angle.
Positive values are for the right looking situation. 
Start is left most point, Stop is right most point, looking in direction of velocity vector.

Optional in the analysis part are:
```
<PolarView>60</PolarView>
<Revisit>True</Revisit>
<Statistic>Mean</Statistic>
<SaveOutput>NetCDF</SaveOutput>
```
- PolarView angle: This parameter can be given to see one part of the globe in an stereographic view, eg.  for the polar region.
  The value describes the minimum bounding latitude visible. When negative the area on the South Pole will be visible.
- Revisit flag: This flag will enable revisit computation after the swath coverage. The statistic will determine
what kind of statistic is displayed per user location.
- SaveOutput: NetCDF or Numpy This flag will enable saving user swath coverage for every timestep.

<img src="/docs/obs_swath_push_broom.png" alt="cov_satellite_push_broom"/>
<img src="/docs/obs_swath_push_broom_revisit.png" alt="cov_satellite_push_broom_revisit"/>


### obs_sza_subsat
Plots the Solar Zenith Angle SZA for the subsatellite point. 
All the satellites defined in the config will be used.

The following parameters are needed:
```
<Analysis>
    <Type>obs_sza_subsat</Type>
</Analysis>
```
<img src="/docs/obs_sza_subsat.png" alt="obs_sza_subsat"/>

Optional in the analysis part are:
```
<PolarView>90</PolarView>
<RangeLatitude>-80,80,10</RangeLatitude>
<SaveOutput>Numpy</SaveOutput>
```
- PolarView angle: This parameter can be given to see one part of the globe in an stereographic view, eg.  for the polar region.
  When negative the area on the South Pole will be visible.
- RangeLatitude: Min_Lat, Max_Lat and Lat_Step in degrees, will enable plots vs. latitude.
- SaveOutput: Will write to file the SZA vs latitude averaged over the simulation time.

<img src="/docs/obs_sza_subsat_lat.png" alt="obs_sza_subsat_lat"/>
<img src="/docs/obs_sza_subsat_lat_year.png" alt="obs_sza_subsat_lat_year"/>    


### com_gr2sp_budget
Plots the link budget parameters for a certain ground station to all satellites. 
The models used are coming from ITU-R python package itur, which implements:
- ITU-R P.453-13: The radio refractive index: its formula and refractivity data
- ITU-R P.618-13: Propagation data and prediction methods required for the design of Earth-space telecommunication systems
- ITU-R P.676-11: Attenuation by atmospheric gases 
- ITU-R P.835-6: Reference Standard Atmospheres 
- ITU-R P.836-6: Water vapour: surface density and total columnar content 
- ITU-R P.837-7: Characteristics of precipitation for propagation modelling 
- ITU-R P.838-3: Specific attenuation model for rain for use in prediction methods 
- ITU-R P.839-4: Rain height model for prediction methods. 
- ITU-R P.840-7: Attenuation due to clouds and fog
- ITU-R P.1144-7: Interpolation methods for the geophysical properties used to compute propagation effects 
- ITU-R P.1511-1: Topography for Earth-to-space propagation modelling 
- ITU-R P.1853-1: Tropospheric attenuation time series synthesis

The following parameters are needed:
```
<Analysis>
    <Type>com_gr2sp_budget</Type>
    <GroundStationID>1</GroundStationID>
    <TransmitterObject>Satellite</TransmitterObject>
    <CarrierFrequency>10e9</CarrierFrequency>
    <TransmitPowerW>10</TransmitPowerW>
    <TransmitLossesdB>2</TransmitLossesdB>
    <TransmitGaindB>20</TransmitGaindB>
    <ReceiveGaindB>64</ReceiveGaindB>
    <ReceiveLossesdB>3</ReceiveLossesdB>
    <ReceiveTempK>290</ReceiveTempK>
    <PExceedPerc>0.01</PExceedPerc>
    <IncludeGas>True</IncludeGas>
    <IncludeRain>True</IncludeRain>
    <IncludeScintillation>False</IncludeScintillation>
    <IncludeClouds>False</IncludeClouds>
</Analysis>
```
Parameters are:
- GroundStationID: Station to be used, refer to the ground segment part.
- TransmitterObject: Satellite or Ground Station, which one is transmitting
- CarrierFrequency: Carrier frequency of signal in Hz
- TransmitPowerW: Transmit power of transmitter in W
- TransmitLossesdB: All transmit losses in dB
- TransmitGaindB: Transmit gain of antenna in dB
- ReceiveGaindB: Receive gain of antenna in dB
- ReceiveLossesdB: All receive losses in dB
- ReceiveTempK: Receive Temperature in K
- PExceedPerc: Probability to exceed attenuation values in %.
- IncludeGas: Whether gas attenuation should be included True/False
- IncludeRain: Whether rain attenuation should be included True/False
- IncludeScintillation: Whether scintillation attenuation should be included True/False
- IncludeClouds: Whether cloud attenuation should be included True/False

Optional in the analysis part are:
```
    <ModulationType>BPSK</ModulationType>
    <BitErrorRate>1e-5</BitErrorRate>
    <DataRateBitPerSec>1e6</DataRateBitPerSec>
```

Some parameters can be entered to get the required CN0:
- ModulationType: 'BPSK' or 'QPSK'
- BitErrorRate: bit error rate required
- DataRateBitPerSec: datarate required

<img src="/docs/com_gr2sp_budget.png" alt="com_gr2sp_budget"/>

### com_gr2sp_budget_interference
Plots the link budget parameters for a certain ground station to all satellites as in com_gr2sp_budget,
but now includes interference from a second satellite following closely the nominal link satellite. 

The following parameters are needed:
```
<Analysis>
    <Type>com_gr2sp_budget</Type>
    <GroundStationID>1</GroundStationID>
    <TransmitterObject>Satellite</TransmitterObject>
    <CarrierFrequency>10e9</CarrierFrequency>
    <BandWidth>675e6</BandWidth>
    <TransmitPowerW>10</TransmitPowerW>
    <TransmitLossesdB>2</TransmitLossesdB>
    <TransmitGaindB>20</TransmitGaindB>
    <TransmitAntennaDiameter>0.25</TransmitAntennaDiameter>
    <ReceiveGaindB>64</ReceiveGaindB>
    <ReceiveAntennaDiameter>6</ReceiveAntennaDiameter>
    <ReceiveLossesdB>3</ReceiveLossesdB>
    <ReceiveTempK>290</ReceiveTempK>
    <PExceedPerc>0.5</PExceedPerc>
    <IncludeGas>True</IncludeGas>
    <IncludeRain>True</IncludeRain>
    <IncludeScintillation>False</IncludeScintillation>
    <IncludeClouds>False</IncludeClouds>
</Analysis>
```
Parameters are:
- GroundStationID: Station to be used, refer to the ground segment part.
- TransmitterObject: Satellite or Ground Station, which one is transmitting
- CarrierFrequency: Carrier frequency of signal in Hz
- BandWidth: Signal bandwith in Hz
- TransmitPowerW: Transmit power of transmitter in W
- TransmitLossesdB: All transmit losses in dB
- TransmitGaindB: Transmit gain of antenna in dB (theoretical pattern assumed), or,
  TransmitGainManualdB: list of tuples in string from 0-180 off boresight gain values
- TransmitAntennaDiameter: Antenna diameter in m (not used to compute Gain)
- ReceiveGaindB: Receive gain of antenna in dB
- ReceiveLossesdB: All receive losses in dB
- ReceiveTempK: Receive Temperature in K
- ReceiveAntennaDiameter: Antenna diameter in m (not used to compute Gain)
- PExceedPerc: Probability to exceed attenuation values in %.

Optional in the analysis part are:
```
    <ModulationType>BPSK</ModulationType>
    <BitErrorRate>1e-5</BitErrorRate>
    <DataRateBitPerSec>1e6</DataRateBitPerSec>
```

Some parameters can be entered to get the required CN0:
- ModulationType: 'BPSK' or 'QPSK'
- BitErrorRate: bit error rate required
- DataRateBitPerSec: datarate required

<img src="/docs/com_gr2sp_budget_interference.png" alt="com_gr2sp_budget_interference"/>


### com_sp2sp_budget
Plots the link budget parameters for a certain satellite to another satellite. 

The following parameters are needed:
```
<Analysis>
    <Type>com_sp2sp_budget</Type>
    <SatelliteID1>1</SatelliteID1>
    <SatelliteID2>2</SatelliteID2>
    <CarrierFrequency>10e9</CarrierFrequency>
    <TransmitPowerW>10</TransmitPowerW>
    <TransmitLossesdB>2</TransmitLossesdB>
    <TransmitGaindB>20</TransmitGaindB>
    <ReceiveGaindB>20</ReceiveGaindB>
    <ReceiveLossesdB>2</ReceiveLossesdB>
    <ReceiveTempK>290</ReceiveTempK>
</Analysis>
```
Parameters are:
- SatelliteID1: Satellite to be used as transmitter.
- SatelliteID2: Satellite to be used as receiver.
- CarrierFrequency: Carrier frequency of signal in Hz
- TransmitPowerW: Transmit power of transmitter in W
- TransmitLossesdB: All transmit losses in dB
- TransmitGaindB: Transmit gain of antenna in dB

Optional in the analysis part are:
```
    <ModulationType>BPSK</ModulationType>
    <BitErrorRate>1e-5</BitErrorRate>
    <DataRateBitPerSec>1e6</DataRateBitPerSec>
```

Some parameters can be entered to get the required CN0:
- ModulationType: 'BPSK' or 'QPSK'
- BitErrorRate: bit error rate required
- DataRateBitPerSec: datarate required

<img src="/docs/com_sp2sp_budget.png" alt="com_sp2sp_budget"/>


### com_doppler
Plots the doppler shift in Hz for the station to satellites. 

The following parameters are needed:
```
<Analysis>
    <Type>com_doppler</Type>
    <StationID>1</StationID>
    <CarrierFrequency>10e9</CarrierFrequency>
</Analysis>
```
Parameters are:
- StationID: station to be selected.
- CarrierFrequency: Carrier frequency of signal in Hz

<img src="/docs/com_doppler.png" alt="com_doppler"/>


### nav_dilution_of_precision
Plots navigation dilution of precision for a user grid. 

The following parameters are needed:
```
<Analysis>
    <Type>nav_dilution_of_precision</Type>
    <Direction>Ver</Direction>
    <Statistic>Max</Statistic>
</Analysis>
```
Parameters are:
- Direction: Hor/Ver/Pos direction of interest.
- Statistic: Min/Mean/Max/Median/Std statistic of interest

<img src="/docs/nav_dilution_of_precision.png" alt="nav_dilution_of_precision"/>


### nav_accuracy
Plots navigation accuracy for a user grid based on on sqrt(uere)*DOP computations. 

The following parameters are needed:
```
<Analysis>
    <Type>nav_dilution_of_precision</Type>
    <Direction>Ver</Direction>
    <Statistic>Max</Statistic>
</Analysis>
```
Parameters are:
- Direction: Hor/Ver/Pos direction of interest.
- Statistic: Min/Mean/Max/Median/Std statistic of interest

Additionally the constellation needs to be supplied with an elevation dependent list of uere values:
```
    <UERE>1.72,1.72,1.17,1.02,0.92,0.92,0.85,0.85,0.81,0.81,0.80,0.80,0.79,0.79,0.79,0.79,0.79,0.79</UERE>
```

<img src="/docs/nav_accuracy.png" alt="nav_accuracy"/>





