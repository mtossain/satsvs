import numpy as np
from math import floor

from constants import pi
import misc_fn


class KeplerSet:

    def __init__(self):
        self.EpochMJD = 0.0
        self.SemiMajorAxis = 0.0
        self.Eccentricity = 0.0
        self.Inclination = 0.0
        self.RAAN = 0.0
        self.ArgOfPerigee = 0.0
        self.MeanAnomaly = 0.0


class Constellation:

    def __init__(self):
        self.ConstellationID = 0.0
        self.ConstellationName = ''
        self.ConstellationType = 0.0
        self.NumberOfPlanes = 0.0
        self.Size = 0.0
        self.TLEFileName = ''  # Could not be used in normal definition


class Satellite:

    def __init__(self):

        self.ConstellationID = 0
        self.ConstellationName = ''
        self.SatelliteID = 0
        self.Plane = 0
        self.Name = ''

        self.AntennaMask = []  # Could be varying over azimuth...
        self.AntennaMaskMaximum = []  # Could be varying over azimuth...

        self.Kepler = KeplerSet()

        self.PosVelECI = 6*[0.0]
        self.PosVelECF = 6*[0.0]
        self.LLA = 3*[0.0]  # For ground track

        self.IdxStationInView = []  # Indices of station which are in view
        self.NumStationInView = 0  # Number of stations that 'see' this satellite

        # int* IdxSatelliteInView  # Indices of satellites which are in view
        self.IdxSatelliteInView = []
        self.NumSatelliteInView = 0  # Number of satellites that 'see' this satellite

        self.Metric = []  # For analysis purposes

    def DeterminePosVelECI(self, MJDRequested):
        self.PosVelECI = misc_fn.KEP2XYZ(MJDRequested, self.Kepler)

    def DeterminePosVelECF(self, GMSTRequested):  # ECF
        self.PosVelECF = misc_fn.SpinVector(-GMSTRequested, self.PosVelECI)  # assume ECI and GMST computed elsewhere

    def DetermineLLA(self):
        self.LLA = misc_fn.XYZ2LLA(self.PosVelECF)


class GroundStation:

    def __init__(self):
        self.ConstellationID = 0
        self.GroundStationID = 0
        self.GroundStationName = ''
        self.ReceiverConstellation = ''
        self.ElevationMask = []  # Could be varying over azimuth...
        self.ElevationMaskMaximum = []  # Could be varying over azimuth...

        self.PosVelECI = 6*[0.0]
        self.PosVelECF = 6*[0.0]
        self.LLA = 3*[0.0]
        self.IdxSatInView = []  # Indices of satellites which are in view
        self.NumSatInView = 0

    def DeterminePosVelECF(self):
        xyz = misc_fn.LLA2XYZ(self.LLA)  # ECF
        self.PosVelECF[0:3] = xyz

    def DeterminePosVelECI(self, GMSTRequested):
        self.PosVelECI = misc_fn.SpinVector(GMSTRequested, self.PosVelECF)
        # TODO Velocity of ground stations, currently set to zero....


class User:

    def __init__(self):

        self.Type = 0
        self.UserID = 0
        self.UserName = ''
        self.ReceiverConstellation = ''
        self.ElevationMask = []  # Could be varying over azimuth...
        self.ElevationMaskMaximum = []  # Could be varying over azimuth...

        self.PosVelECI = 6*[0.0]
        self.PosVelECF = 6*[0.0]
        self.LLA = 3*[0.0]
        self.NumLat = 0
        self.NumLon = 0

        self.IdxSatInView = []  # Indices of satellites which are in view
        self.NumSatInView = 0

        self.TLEFileName = ''  # In case user is a spacecraft
        self.Kepler = KeplerSet()  # In case user is a spacecraft

        self.Metric = []  # For analysis purposes

    def DeterminePosVelECF(self):
        xyz = misc_fn.LLA2XYZ(self.LLA)
        self.PosVelECF[0:3] = xyz  # ECF

    def DeterminePosVelECI(self, GMSTRequested):
        # Compute ECI coordinates from ECF set and GMST
        # Requires the ECF to be computed first
        self.PosVelECI = misc_fn.SpinVector(GMSTRequested, self.PosVelECF)  # ECI

    def DeterminePosVelTLE(self, GMSTRequested, MJDRequested):  # For spacecraft user
        # Compute ECF and ECI coordinates from MJD and TLE set
        self.PosVelECI = misc_fn.KEP2XYZ(MJDRequested, self.Kepler)  # ECI
        self.PosVelECF = misc_fn.SpinVector(-GMSTRequested, self.PosVelECI)  # ECF


class Ground2SpaceLink:

    def __init__(self):

        self.NumSat = 0  # for indexing static variable
        self.LinkInUse = True  # Defines whether ground asset is using this satellite (from ReceiverConstellation)

        self.Azimuth = 0.0  # Radians
        self.Elevation = 0.0  # Radians
        self.Azimuth2 = 0.0  # Receiver as seen from satellite
        self.Elevation2 = 0.0  # Receiver as seen from satellite (equivalent to off-nadir from satellite)

        self.Ground2SpaceECF = [0.0,0.0,0.0]  # m
        self.Distance = 0.0  # m

        self.GroundStation = GroundStation()
        self.Satellite = Satellite()

        self.Metric = []  # For analysis purposes

    def ComputeLinkGroundStation(self, station, satellite):

        # For reasons of speed this function is coded inline, since this function is done for every time step,
        # for every combination of user and satellite. Also the computation of azimuth is not done standard,
        # since it takes more time and not needed by all analysis

        self.Ground2SpaceECF = [satellite.PosVelECF[i] - station.PosVelECF[i] for i in range(3)]

        self.Distance = np.linalg.norm(self.Ground2SpaceECF)

        self.Azimuth, self.Elevation = misc_fn.CalcAzEl(satellite.PosVelECF, station.PosVelECF)  # From Station to Satellite
        self.Azimuth2, self.Elevation2 = misc_fn.CalcAzEl(station.PosVelECF, satellite.PosVelECF)  # From Satellite to Station

        self.GroundStation = station
        self.Satellite = satellite


    def CheckMaskingStation(self, station):
        # Check whether satellite is above masking angle defined for station/user
        # Optionally also a maximum mask angle is defined

        in_view = False
        number_of_masks = len(station.ElevationMask)

        if number_of_masks == 1:
            if station.ElevationMask[0] < self.Elevation < station.ElevationMaskMaximum[0]:
                in_view = True
        else:  # More than one mask
            az_cake_piece_angle = 2 * pi / number_of_masks
            idx_az = int(floor(self.Azimuth / az_cake_piece_angle))
            if station.ElevationMask[idx_az] < self.Elevation < station.ElevationMaskMaximum[idx_az]:
                in_view = True

        return in_view


class User2SpaceLink:

    def __init__(self):

        self.NumSat = 0  # for indexing static variable
        self.LinkInUse = True  # Defines whether ground asset is using this satellite (from ReceiverConstellation)

        self.Azimuth = 0.0  # Radians
        self.Elevation = 0.0  # Radians
        self.Azimuth2 = 0.0  # Receiver as seen from satellite
        self.Elevation2 = 0.0  # Receiver as seen from satellite (equivalent to off-nadir from satellite)

        self.User2SpaceECF = [0.0,0.0,0.0]  # m
        self.Distance = 0.0  # m

        self.Metric = []  # For analysis purposes

    def ComputeLinkUser(self, user, satellite):

        self.User2SpaceECF = [satellite.PosVelECF[i] - user.PosVelECF[i] for i in range(3)]

        self.Distance = np.linalg.norm(self.User2SpaceECF)

        self.Azimuth, self.Elevation = misc_fn.CalcAzEl(satellite.PosVelECF, user.PosVelECF)

        # TODO self.User should be defined?

    def CheckMaskingUser(self, user):
        # Check whether satellite is above masking angle defined for station/user
        # Optionally also a maximum mask angle is defined

        in_view = False
        number_of_masks = len(user.ElevationMask)

        if number_of_masks == 1:
            if user.ElevationMask[0] < self.Elevation < user.ElevationMaskMaximum[0]:
                in_view = True
        else:  # More than one mask
            az_cake_piece_angle = 2 * pi / number_of_masks
            idx_az = int(floor(self.Azimuth / az_cake_piece_angle))
            if user.ElevationMask[idx_az] < self.Elevation < user.ElevationMaskMaximum[idx_az]:
                in_view = True

        return in_view


class Space2SpaceLink:

    def __init__(self):

        self.LinkInUse = True  # Defines whether ground asset is using this satellite (from ReceiverConstellation)

        self.IdxSat1 = 0  # First satellite in the link
        self.IdxSat2 = 0  # Second satellite in the link

        self.IdxSatTransm = 0  # Pointer to transmitting sat
        self.IdxSatRecv = 0  # Pointer to receiving sat

        self.AzimuthTransm = 0.0  # Radians
        self.ElevationTransm = 0.0  # Radians (is really the LOS-NADIR angle from one sat to the other)
        self.AzimuthRecv = 0.0  # Radians
        self.ElevationRecv = 0.0  # Radians

        self.Space2SpaceECF = 3*[0]
        self.Distance = 0  # m

        self.Metric = []  # For analysis purposes

    def ComputeLink(self, satellite1, satellite2):

        # For reasons of speed this function is coded inline, since this function is done for every time step,
        # for every combination of user and satellite. Also the computation of azimuth is not done standard,
        # since it takes more time and not needed by all analysis

        self.Space2SpaceECF = [satellite2.PosVelECF[i] - satellite1.PosVelECF[i] for i in range(3)]

        self.Distance = np.linalg.norm(self.Space2SpaceECF)

        self.AzimuthTransm, self.ElevationTransm = misc_fn.CalcAzEl(satellite1.PosVelECF, satellite2.PosVelECF)
        self.AzimuthRecv, self.ElevationRecv = misc_fn.CalcAzEl(satellite2.PosVelECF, satellite1.PosVelECF)

        return 1

    def CheckMasking(self, satelliteTransm, satelliteRecv):

        # Check whether satellite is above masking angle defined for station/user
        # Optionally also a maximum mask angle is defined

        in_view_tx = False
        in_view_rx = False

        num_of_masks_tx = len(satelliteTransm.AntennaMask)  # If only one value
        num_of_masks_rx = len(satelliteRecv.AntennaMask)  # If only one value

        # if satellite_tx and satellite_rx are identical
        if self.Distance == 0.0:
            return False

        if num_of_masks_tx == 0:  # omnidirectional antenna
            in_view_tx = True

        if num_of_masks_tx == 1:
            # checking for mask of satellite 1
            if satelliteTransm.AntennaMask[0] < self.ElevationTransm < satelliteTransm.AntennaMaskMaximum[0]:
                in_view_tx = True

        else:  # More than one mask
            az_cake_piece_angle = 2 * pi / num_of_masks_tx
            idx_az = int(floor(self.AzimuthTransm / az_cake_piece_angle))
            if satelliteTransm.AntennaMask[idx_az] < self.ElevationTransm < satelliteTransm.AntennaMaskMaximum[idx_az]:
                in_view_tx = True

        if num_of_masks_rx == 0: # omnidirectional antenna
            in_view_rx = True

        if num_of_masks_rx == 1:
            # checking for mask of satellite 2
            if satelliteRecv.AntennaMask[0] < self.ElevationRecv < satelliteRecv.AntennaMaskMaximum[0]:
                in_view_rx = True

        else:  # More than one mask
            az_cake_piece_angle = 2 * pi / num_of_masks_rx
            idx_az = int(floor(self.AzimuthRecv / az_cake_piece_angle))
            if satelliteRecv.AntennaMask[idx_az] < self.ElevationRecv < satelliteRecv.AntennaMaskMaximum[idx_az]:
                in_view_rx = True

        return in_view_tx, in_view_rx
