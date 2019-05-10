
import misc_fn
from math import sqrt, floor
from constants import pi


class KeplerSet:

    def __init__(self):
        self.EpochMJD = 0
        self.SemiMajorAxis = 0
        self.Eccentricity = 0
        self.Inclination = 0
        self.RAAN = 0
        self.ArgOfPerigee = 0
        self.MeanAnomaly = 0


class Constellation:

    def __init__(self):
        self.ConstellationID = 0
        self.ConstellationName = ''
        self.ConstellationType = 0
        self.NumberOfPlanes = 0
        self.Size = 0
        self.TLEFileName = ''  # Could not be used in normal definition
        self.SP3FileName= ''


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

        self.PosVelECI = [0,0,0,0,0,0]
        self.PosVelECF = [0,0,0,0,0,0]
        self.LLA = [0,0,0]  # For ground track

        self.IdxStationInView = []  # Indices of station which are in view
        self.NumStationInView = 0  # Number of stations that 'see' this satellite

        # int* IdxSatelliteInView  # Indices of satellites which are in view
        self.IdxSatelliteInView = []
        self.NumSatelliteInView = 0  # Number of satellites that 'see' this satellite

        self.Metric = []  # For analysis purposes

    def DeterminePosVelECI(self, MJDRequested):
        self.PosVelECI = misc_fn.KEP2XYZ(MJDRequested, self.Kepler)

    def DeterminePosVelECF(self, GMSTRequested):  # ECF
        self.PosVelECF = misc_fn.SpinVector(-GMSTRequested, self.PosVelECI) # assume ECI and GMST computed elsewhere

    def DeterminePosVelLLA(self):
        self.LLA = misc_fn.XYZ2LLA(self.PosVelECF)

    # operator==(Satellite const& rhs)  //Operator overloading


class GroundStation:

    def __init__(self):
        self.ConstellationID = 0
        self.GroundStationID = 0
        self.GroundStationName = ''
        self.ReceiverConstellation = ''
        self.ElevationMask = []  # Could be varying over azimuth...
        self.ElevationMaskMaximum = []  # Could be varying over azimuth...

        self.PosVelECI = [0,0,0,0,0,0]
        self.PosVelECF = [0,0,0,0,0,0]
        self.LLA = [0,0,0]
        self.IdxSatInView = []  # Indices of satellites which are in view
        self.NumSatInView = 0

    def DeterminePosVelECI(self, GMSTRequested):
        self.PosVelECI = misc_fn.SpinVector(GMSTRequested, self.PosVelECF)
        # To be done:
        # Velocity of ground stations, currently set to zero....

    def DeterminePosVelECF(self):

        XYZ = misc_fn.LLA2XYZ(self.LLA)  # ECF
        self.PosVelECF[0] = XYZ[0]
        self.PosVelECF[1] = XYZ[1]
        self.PosVelECF[2] = XYZ[2]
        self.PosVelECF[3] = 0
        self.PosVelECF[4] = 0
        self.PosVelECF[5] = 0


class User:

    def __init__(self):

        self.Type = 0
        self.UserID = 0
        self.UserName = ''
        self.ReceiverConstellation = ''
        self.ElevationMask = []  # Could be varying over azimuth...
        self.ElevationMaskMaximum = []  # Could be varying over azimuth...

        self.PosVelECI = [0,0,0,0,0,0]
        self.PosVelECF = [0,0,0,0,0,0]
        self.LLA = [0,0,0]
        self.NumLat = 0
        self.NumLon = 0

        self.IdxSatInView = []  # Indices of satellites which are in view
        self.NumSatInView = 0

        self.TLEFileName = ''  # In case user is a spacecraft
        self.Kepler = KeplerSet()  # In case user is a spacecraft

        self.Metric = [] # For analysis purposes

    def DeterminePosVelECF(self):
        self.PosVelECF = misc_fn.LLA2XYZ(self.LLA)  # ECF

    def DeterminePosVelECI(self, GMSTRequested):
        # Compute ECI coordinates from ECF set and GMST
        # Requires the ECF to be computed first
        self.PosVelECI = misc_fn.SpinVector(GMSTRequested, self.PosVelECI)  # ECI

    def DeterminePosVelTLE(self, GMSTRequested, MJDRequested):  # For spacecraft user
        # Compute ECF and ECI coordinates from MJD and TLE set
        self.PosVelECI = misc_fn.KEP2XYZ(MJDRequested, self.Kepler)  # ECI
        self.PosVelECF = misc_fn.SpinVector(-GMSTRequested, self.PosVelECI)  # ECF


class Ground2SpaceLink:

    def __init__(self):

        self.NumSat = 0  # for indexing static variable
        self.LinkInUse = True  # Defines whether ground asset is using this satellite (from ReceiverConstellation)

        self.Azimuth = 0  # Radians
        self.Elevation = 0  # Radians
        self.Azimuth2 = 0  # Receiver as seen from satellite
        self.Elevation2 = 0  # Receiver as seen from satellite (equivalent to off-nadir from satellite)

        self.Ground2SpaceECF = [0,0,0]  # m
        self.Distance = 0  # m

        self.GroundStation = GroundStation()
        self.Satellite = Satellite()

        self.Metric = []  # For analysis purposes

    def ComputeLinkGroundStation(self, station, satellite):

        #For reasons of speed this function is coded inline, since this function is done for every time step,
        #for every combination of user and satellite. Also the computation of azimuth is not done standard,
        #since it takes more time and not needed by all analysis

        for i in range(3):
            self.Ground2SpaceECF[i] = satellite.PosVelECF[i] - station.PosVelECF[i]
        self.Distance = sqrt(self.Ground2SpaceECF[0] * self.Ground2SpaceECF[0] + self.Ground2SpaceECF[1] * \
                             self.Ground2SpaceECF[1] + self.Ground2SpaceECF[2] * self.Ground2SpaceECF[2])

        self.Azimuth, self.Elevation = misc_fn.CalcAzEl(satellite.PosVelECF, station.PosVelECF)  # From Station to Satellite
        self.Azimuth2, self.Elevation2 = misc_fn.CalcAzEl(station.PosVelECF, satellite.PosVelECF)  # From Satellite to Station

        self.GroundStation = station
        self.Satellite = satellite


    def ComputeLinkUser(self, user, satellite):

        for i in range(3):
            self.Ground2SpaceECF[i] = satellite.PosVelECF[i] - user.PosVelECF[i]
        self.Distance = sqrt(self.Ground2SpaceECF[0] * self.Ground2SpaceECF[0] + self.Ground2SpaceECF[1] * \
                        self.Ground2SpaceECF[1] + self.Ground2SpaceECF[2] * self.Ground2SpaceECF[2])

        self.Azimuth, self.Elevation = misc_fn.CalcAzEl(satellite.PosVelECF, user.PosVelECF)

    def CheckMasking(self, station):
        # Check whether satellite is above masking angle defined for station/user
        # Optionally also a maximium mask angle is defined

        NumberOfMasks = len(station.ElevationMask)

        if NumberOfMasks == 1:
            if self.Elevation > station.ElevationMask[0] and self.Elevation < station.ElevationMaskMaximum[0]:
                InView = True
        else:  # More than one mask
            AzCakePieceAngle = 2 * pi / NumberOfMasks
            IdxAzimuth = int(floor(self.Azimuth / AzCakePieceAngle))
            if self.Elevation > station.ElevationMask[IdxAzimuth] and self.Elevation < station.ElevationMaskMaximum[IdxAzimuth]:
                InView = True

        return InView

    def CheckMasking(self, user):
        # Check whether satellite is above masking angle defined for station/user
        # Optionally also a maximium mask angle is defined

        InView = False
        NumberOfMasks = len(user.ElevationMask)

        if NumberOfMasks == 1:
            if self.Elevation > user.ElevationMask[0] and self.Elevation < user.ElevationMaskMaximum[0]:
                InView = True
        else:  # More than one mask
            AzCakePieceAngle = 2 * pi / NumberOfMasks
            IdxAzimuth = int(floor(self.Azimuth / AzCakePieceAngle))
            if self.Elevation > user.ElevationMask[IdxAzimuth] and self.Elevation < user.ElevationMaskMaximum[IdxAzimuth]:
                InView = True

        return InView

    #Ground2SpaceLink(const Ground2SpaceLink &p) # Copy constructor
    #Ground2SpaceLink()


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

        self.Space2SpaceECF = [0, 0, 0]
        self.Distance = 0  # m

        self.Metric = []  # For analysis purposes

    """def ComputeLink(self):

        if self.IdxSatTransm == self.IdxSatRecv: # avoid calculating a link between two identical sats
            return 0

        # For reasons of speed this function is coded inline, since this function is done for every time step,
        # for every combination of user and satellite. Also the computation of azimuth is not done standard,
        # since it takes more time and not needed by all analysis

        for i in range(3):
            self.Space2SpaceECF[i] = self.IdxSatRecv->PosVelECF[i] - self.IdxSatTransm->PosVelECF[i]
        self.Distance = sqrt(self.Space2SpaceECF[0] * self.Space2SpaceECF[0] + self.Space2SpaceECF[1] * \
                             self.Space2SpaceECF[1] + self.Space2SpaceECF[2] * self.Space2SpaceECF[2])

        self.AzimuthTransm, self.ElevationTransm = misc_fn.CalcAzEl(self.IdxSatTransm->PosVelECF, self.IdxSatRecv->PosVelECF)

        self.AzimuthRecv, self.ElevationRecv = misc_fn.CalcAzEl(self.IdxSatRecv->PosVelECF, self.IdxSatTransm->PosVelECF)

        return 1
    """


    def ComputeLink(self, satellite1, satellite2):

        if satellite1 == satellite2:  # avoid calculating a link between two identical satellites
            return 0

        # For reasons of speed this function is coded inline, since this function is done for every time step,
        # for every combination of user and satellite. Also the computation of azimuth is not done standard,
        # since it takes more time and not needed by all analysis

        for i in range(3):
            self.Space2SpaceECF[i] = satellite2.PosVelECF[i] - satellite1.PosVelECF[i]
        self.Distance = sqrt(self.Space2SpaceECF[0] * self.Space2SpaceECF[0] + self.Space2SpaceECF[1] *
                        self.Space2SpaceECF[1] + self.Space2SpaceECF[2] * self.Space2SpaceECF[2])

        self.AzimuthTransm, self.ElevationTransm = misc_fn.CalcAzEl(satellite1.PosVelECF, satellite2.PosVelECF)

        self.AzimuthRecv, self.ElevationRecv  = misc_fn.CalcAzEl(satellite2.PosVelECF, satellite1.PosVelECF)

        return 1


    def CheckMasking(self, satelliteTransm, satelliteRecv):

        # Check whether satellite is above masking angle defined for station/user
        # Optionally also a maximium mask angle is defined

        InViewTransm = False
        InViewRecv = False

        NumberOfMasksTransm = len(satelliteTransm.AntennaMask) # If only one value
        NumberOfMasksRecv = len(satelliteRecv.AntennaMask) # If only one value

        # if satelliteTransm and satelliteRecv are identical
        if self.Distance == 0.0:
            return False

        if NumberOfMasksTransm == 0:  # omnidirectional antenna
            InViewTransm = True

        if NumberOfMasksTransm == 1:
            # checking for mask of satellite 1
            if self.ElevationTransm >= satelliteTransm.AntennaMask[0] and \
                    self.ElevationTransm <= satelliteTransm.AntennaMaskMaximum[0]:
                InViewTransm = True

        else:  # More than one mask
            AzCakePieceAngle = 2 * pi / NumberOfMasksTransm
            IdxAzimuth = int(floor(self.AzimuthTransm / AzCakePieceAngle))
            if self.ElevationTransm >= satelliteTransm.AntennaMask[IdxAzimuth] and \
                    self.ElevationTransm <= satelliteTransm.AntennaMaskMaximum[IdxAzimuth]:
                InViewTransm = True

        if NumberOfMasksRecv == 0: # omnidirectional antenna
            InViewRecv = True

        if NumberOfMasksRecv == 1:
            # checking for mask of satellite 2
            if self.ElevationRecv >= satelliteRecv.AntennaMask[0] and \
                    self.ElevationRecv <= satelliteRecv.AntennaMaskMaximum[0]:
                InViewRecv = True

        else:  # More than one mask
            AzCakePieceAngle = 2 * pi / NumberOfMasksRecv
            IdxAzimuth = int(floor(self.AzimuthRecv / AzCakePieceAngle))
            if self.ElevationRecv >= satelliteRecv.AntennaMask[IdxAzimuth] and \
                    self.ElevationRecv <= satelliteRecv.AntennaMaskMaximum[IdxAzimuth]:
                InViewRecv = True

        return InViewTransm, InViewRecv

    #Space2SpaceLink(const Space2SpaceLink& p)
    #Space2SpaceLink()

