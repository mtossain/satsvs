import xml.etree.ElementTree as ET
from math import floor
from astropy.time import Time

import misc_fn
from constants import *
from analysis import Analysis
from segments import Constellation, Satellite, GroundStation, User, Ground2SpaceLink, Space2SpaceLink

import logging
import logging.handlers as handlers

logger = logging.getLogger('config')
logger.setLevel(logging.INFO)
logHandler = handlers.RotatingFileHandler('main.log', maxBytes=5*1024*1024)
logHandler.setLevel(logging.INFO)
logHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(logHandler)


class AppConfig:
    
    def __init__(self):

        self.ConstellationList = []
        self.SatelliteList = []
        self.GroundStationList = []
        self.UserList = []
        self.AnalysisList = []
        self.GroundStation2SatelliteList = []
        self.User2SatelliteList = []
        self.Satellite2SatelliteList = []
        self.RadiationPropList = []

        self.StartDateTime = 0  # MJD
        self.StopDateTime = 0  # MJD
        self.TimeStep = 0  # s
        self.NumTimeSteps = 0
        self.NumEpoch = 0
        self.NumConstellation = 0
        self.NumSat = 0
        self.NumGroundStation = 0
        self.NumUser = 0
        self.NumSpace2Space = 0
        self.NumGround2Space = 0

        # Boolean regarding HPOP ( if false no perturbations are taken into account)
        self.HPOP = True
        # Propagation parameters
        self.mass = 0
        self.solar_Area = 0
         # Forces
        self.EarthGravity = True
        self.PlanetsPerturbations = True
        self.SRP = True
        self.SolidTides = True
        self.OceanTides = True
        self.Relativity = True
        self.Albedo = True
        self.Empirical = True

         # EGM
        self.Degree = 0
        self.Order = 0
        self.Model = ''
         # Planets Perturbations
        self. Sun = True
        self.Moon = True
        self.Mercury = True
        self.Venus = True
        self. Mars = True
        self.Jupiter = True
        self.Saturn = True
        self.Uranus = True
        self.Neptune = True

        # Numerical Integration
        self.TimeStepPropagation = 0.0
        self.Method = ''

    def LoadSpaceSegment(self, FileName):

        # Get the list of constellations
        tree = ET.parse(FileName)
        root = tree.getroot()
        for constellation in root.iter('Constellation'):
            const = Constellation()
            const.NumberOfPlanes = int(constellation.find('NumOfPlanes').text)
            const.ConstellationID = int(constellation.find('ConstellationID').text)
            const.NumOfSatellites = int(constellation.find('NumOfSatellites').text)
            const.NumberOfPlanes = int(constellation.find('NumOfPlanes').text)
            const.ConstellationName = constellation.find('ConstellationName').text
            logger.info(const.__dict__)
            self.ConstellationList.append(const)
            for satellite in constellation.iter('Satellite'):
                sat = Satellite()
                sat.SatelliteID = int(satellite.find('SatelliteID').text)
                sat.Plane = int(satellite.find('Plane').text)
                sat.ConstellationID = const.ConstellationID
                sat.ConstellationName = const.ConstellationName
                sat.Kepler.EpochMJD = float(satellite.find('EpochMJD').text)
                sat.Kepler.SemiMajorAxis = float(satellite.find('SemiMajorAxis').text)
                sat.Kepler.Eccentricity = float(satellite.find('Eccentricity').text)
                sat.Kepler.Inclination = float(satellite.find('Inclination').text) / 180 * M_PI
                sat.Kepler.RAAN = float(satellite.find('RAAN').text) / 180 * M_PI
                sat.Kepler.ArgOfPerigee = float(satellite.find('ArgOfPerigee').text) / 180 * M_PI
                sat.Kepler.MeanAnomaly = float(satellite.find('MeanAnomaly').text) / 180 * M_PI
                logger.info(sat.__dict__)
                self.SatelliteList.append(sat)
            for item in constellation.iter('TLEFileName'):
                with open(item.text) as file:         
                    cnt = 0
                    for line in file:
                        if cnt == 0:
                            sat = Satellite()
                            sat.Name = line
                        if cnt == 1:
                            sat.SatelliteID = int(line[2:7])  # NORAD Catalog Number
                            sat.ConstellationID = const.ConstellationID
                            sat.ConstellationName = const.ConstellationName
                            fullyear = "20" + line[18:20]
                            epoch_yr = int(fullyear)
                            epoch_doy = float(line[20:31])
                            sat.Kepler.EpochMJD = misc_fn.YYYYDOY2MJD(epoch_yr, epoch_doy)
                        if cnt == 2:
                            mu = GM_Earth
                            mean_motion = float(line[52:59]) / 86400 * 2 * math.pi  # rad / s
                            sat.Kepler.SemiMajorAxis = pow((mu / (pow(mean_motion, 2))), (1.0 / 3.0))
                            eccentricity = "0." + line[26:33]
                            sat.Kepler.Eccentricity = float(eccentricity)
                            sat.Kepler.Inclination = float(line[8:15]) / 180 * math.pi
                            sat.Kepler.RAAN = float(line[17:24]) / 180 * math.pi
                            sat.Kepler.ArgOfPerigee = float(line[34:41]) / 180 * math.pi
                            sat.Kepler.MeanAnomaly = float(line[43:50]) / 180 * math.pi
                            logger.info(['Found satellite:', str(sat.SatelliteID),'Kepler Elements', str(sat.Kepler.__dict__)])
                            cnt=-1
                        cnt += 1
        self.NumSat = len(self.SatelliteList)
        self.NumConstellation = len(self.ConstellationList)

    def LoadGroundSegment(self, FileName):
        # Get the list of constellations
        tree = ET.parse(FileName)
        root = tree.getroot()
        for groundstation in root.iter('GroundStation'):
            gs = GroundStation()
            gs.ConstellationID = int(groundstation.find('ConstellationID').text)
            gs.GroundStationID = int(groundstation.find('GroundStationID').text)
            gs.GroundStationName = groundstation.find('GroundStationName').text
            gs.LLA[0] = float(groundstation.find('Latitude').text) / 180 * pi
            gs.LLA[1] = float(groundstation.find('Longitude').text) / 180 * pi
            gs.LLA[2] = float(groundstation.find('Height').text)
            gs.ReceiverConstellation = groundstation.find('ReceiverConstellation').text
            gs.IdxSatInView = self.NumSat*[0]

            MaskValues = groundstation.find('ElevationMask').text.split(',')
            for MaskValue in MaskValues:
                gs.ElevationMask.append(float(MaskValue) / 180 * M_PI)
            if groundstation.find('ElevationMaskMaximum') is not None: # Optional
                MaskMaxValues = groundstation.find('ElevationMaskMaximum').text.split(',')
                for MaskMaxValue in MaskMaxValues:
                    gs.ElevationMaskMaximum.append(MaskMaxValue / 180 * M_PI)
            else:  # If no maximum set to 90 degrees
                for MaskValue in MaskValues:
                    gs.ElevationMaskMaximum.append(90.0 / 180.0 * M_PI)

            gs.DeterminePosVelECF()
            logger.info(gs.__dict__)
            self.GroundStationList.append(gs)

        self.NumGroundStation = len(self.GroundStationList)


    def LoadUserSegment(self, FileName):
        # Get the list of constellations
        tree = ET.parse(FileName)
        root = tree.getroot()
        for userelement in root.iter('UserSegment'):

            if userelement.find('Type').text == 'Static':
                user = User()
                user.Type = userelement.find('Type').text
                user.UserID = 0
                user.LLA[0] = float(userelement.find('Latitude').text) / 180 * pi
                user.LLA[1] = float(userelement.find('Longitude').text) / 180 * pi
                user.LLA[2] = float(userelement.find('Height').text)
                user.ReceiverConstellation = userelement.find('ReceiverConstellation').text
                user.IdxSatInView = self.NumSat*[0]
                MaskValues = userelement.find('ElevationMask').text.split(',')
                user.ElevationMask = list(map(lambda x: float(x) / 180 * M_PI, MaskValues))
                if userelement.find('ElevationMaskMaximum') is not None:
                    user.ElevationMaskMaximum = list(map(lambda x: float(x) / 180 * M_PI,
                                                    (userelement.find('ElevationMaskMaximum').text.split(','))))
                else:
                    user.ElevationMaskMaximum = len(MaskValues) * [90.0 / 180.0 * M_PI]
                user.DeterminePosVelECF()
                self.UserList.append(user)

            if userelement.find('Type').text == 'Grid':
                LatMin = float(userelement.find('LatMin').text)
                LatMax = float(userelement.find('LatMax').text)
                LonMin = float(userelement.find('LonMin').text)
                LonMax = float(userelement.find('LonMax').text)
                LatStep = float(userelement.find('LatStep').text)
                LonStep = float(userelement.find('LonStep').text)
                NumLat = (int) ((LatMax - LatMin) / LatStep) + 1
                NumLon = (int) ((LonMax - LonMin) / LonStep) + 1
                Height = float(userelement.find('Height').text)
                ReceiverConstellation = userelement.find('ReceiverConstellation').text
                MaskValues = userelement.find('ElevationMask').text.split(',')
                MaskMaximumumValues = []
                if userelement.find('ElevationMaskMaximum') is not None:
                    MaskMaximumumValues = userelement.find('ElevationMaskMaximum').text.split(',')

                # Now make the full list of users
                CntUsers = 0
                for i in range(NumLat):
                    Latitude = LatMin + LatStep * i
                    for j in range(NumLon):
                        Longitude = LonMin + LonStep * j
                        user = User()
                        user.Type = "Grid"
                        user.UserID = CntUsers
                        user.UserName = "Grid"
                        user.LLA[0] = Latitude / 180 * M_PI
                        user.LLA[1] = Longitude / 180 * M_PI
                        user.LLA[2] = Height
                        user.ReceiverConstellation = ReceiverConstellation
                        user.ElevationMask = list(map(lambda x: float(x)/180*M_PI, MaskValues))
                        if userelement.find('ElevationMaskMaximum') is not None:
                            user.ElevationMaskMaximum = list(map(lambda x: float(x)/180*M_PI, MaskMaximumumValues))
                        else:
                            user.ElevationMaskMaximum = len(MaskValues)*[90.0 / 180.0 * M_PI]

                        user.DeterminePosVelECF()  # Do it once to initialise
                        user.IdxSatInView = self.NumSat*[0]
                        self.UserList.append(user)
                        CntUsers += 1

            if userelement.find('Type').text == 'Spacecraft': # TBD Only one for the moment... Only defined by TLE for the mpment
                user = User()
                user.UserID = 0
                user.Type = 'Spacecraft'
                user.ReceiverConstellation = userelement.find('ReceiverConstellation').text
                user.IdxSatInView = self.NumSat*[0]
                MaskValues = userelement.find('ElevationMask').text.split(',')
                user.ElevationMask = list(map(lambda x: float(x) / 180 * M_PI, MaskValues))
                if userelement.find('ElevationMaskMaximum') is not None:
                    user.ElevationMaskMaximum = list(map(lambda x: float(x) / 180 * M_PI,
                                                    (userelement.find('ElevationMaskMaximum').text.split(','))))
                else:
                    user.ElevationMaskMaximum = len(MaskValues) * [90.0 / 180.0 * M_PI]

                user.TLEFileName = userelement.find('TLEFileName').text
                with open(user.TLEFileName) as file:
                    cnt = 0
                    for line in file:
                        if cnt == 0:
                            user.UserName = line
                        if cnt == 1:
                            user.UserID = int(line[2:7])  # NORAD Catalog Number
                            fullyear = "20" + line[18:20]
                            epoch_yr = int(fullyear)
                            epoch_doy = float(line[20:31])
                            user.Kepler.EpochMJD = misc_fn.YYYYDOY2MJD(epoch_yr, epoch_doy)
                        if cnt == 2:
                            mu = GM_Earth
                            mean_motion = float(line[52:59]) / 86400 * 2 * math.pi  # rad / s
                            user.Kepler.SemiMajorAxis = pow((mu / (pow(mean_motion, 2))), (1.0 / 3.0))
                            eccentricity = "0." + line[26:33]
                            user.Kepler.Eccentricity = float(eccentricity)
                            user.Kepler.Inclination = float(line[8:15]) / 180 * math.pi
                            user.Kepler.RAAN = float(line[17:24]) / 180 * math.pi
                            user.Kepler.ArgOfPerigee = float(line[34:41]) / 180 * math.pi
                            user.Kepler.MeanAnomaly = float(line[43:50]) / 180 * math.pi
                            logger.info(['Found user satellite:', str(user.UserID), 'Kepler Elements', str(user.Kepler.__dict__)])
                            cnt = -1
                        cnt += 1
                GMSTRequested = misc_fn.MJD2GMST(user.Kepler.EpochMJD)
                user.DeterminePosVelTLE(GMSTRequested, user.Kepler.EpochMJD)
                user.DeterminePosVelECF()
                self.UserList.append(user)
        logger.info(user.__dict__)
        self.NumUser = len(self.UserList)

    def SetupGround2Space(self):

        Gr2Sp = Ground2SpaceLink()
        Sp2Sp = Space2SpaceLink()

        CntSpaceLink = 0
        CntGrndLink = 0
        CntUsrLink = 0

        Ground2SpaceLink.NumSat = self.NumSat;
        for i in range (self.NumGroundStation):
            for j in range(self.NumSat):
                self.GroundStation2SatelliteList.append(Gr2Sp)
                if self.GroundStationList[i].ReceiverConstellation[self.SatelliteList[j].ConstellationID - 1:self.SatelliteList[j].ConstellationID] == "1":
                    self.GroundStation2SatelliteList[CntGrndLink].LinkInUse = True
                else:
                    self.GroundStation2SatelliteList[CntGrndLink].LinkInUse = False
                CntGrndLink += 1
        logger.info(["Loaded: ", CntGrndLink , " number of Ground Station to Spacecraft links"])

        for i in range(self.NumUser):
            for j in range(self.NumSat):
                self.User2SatelliteList.append(Gr2Sp)
                if self.UserList[i].ReceiverConstellation[(self.SatelliteList[j].ConstellationID - 1): \
                        self.SatelliteList[j].ConstellationID] == "1":
                    self.User2SatelliteList[CntUsrLink].LinkInUse = True
                else:
                    self.User2SatelliteList[CntUsrLink].LinkInUse = False
                CntUsrLink += 1
        logger.info(["Loaded: ", CntUsrLink , " number of User to Spacecraft links"])

        for i in range(self.NumSat):
            for j in range(self.NumSat):
                # if(i != j){ // avoid link to itself
                Sp2Sp.LinkInUse = True  #TBD Receiver Constellation
                Sp2Sp.IdxSat1 = i
                Sp2Sp.IdxSat2 = j
                Sp2Sp.IdxSatTransm = self.SatelliteList[i]
                Sp2Sp.IdxSatRecv = self.SatelliteList[j]
                self.Satellite2SatelliteList.append(Sp2Sp)
                CntSpaceLink += 1
        logger.info(["Loaded: ", CntSpaceLink , " number of Spacecraft to Spacecraft links"])


        # Allocate the memory for the lists of links
        for j in range(self.NumSat):
            self.SatelliteList[j].IdxStationInView = self.NumGroundStation*[0]
        for j in range(self.NumUser):
            self.UserList[j].IdxSatInView =  self.NumSat*[0]

    def LoadPropagation(self, FileName):
        pass

    def LoadSimulation(self, FileName):
        # Get the list of constellations
        tree = ET.parse(FileName)
        root = tree.getroot()
        for sim in root.iter('SimulationManager'):
            self.StartDateTime = Time(sim.find('StartDate').text,scale='utc').mjd
            self.StopDateTime = Time(sim.find('StopDate').text, scale='utc').mjd
            self.TimeStep = int(sim.find('TimeStep').text)
            self.HPOP = eval(sim.find('HPOP').text)
            logger.info(['Loaded simulation, start:',str(self.StartDateTime),'stop:',str(self.StopDateTime),'TimeStep',str(self.TimeStep)])

            for analysis_conf in sim.iter('Analysis'):
                analysis_obj = Analysis()
                analysis_obj.Type = int(analysis_conf.find('Type').text)
                # Prepare output file
                if analysis_conf.find('Statistic') is not None:
                    analysis_obj.Statistic = analysis_conf.find('Statistic').text
                    out = "Analysis_" + analysis_obj.Type + "_" + analysis_obj.Statistic + ".txt"
                else:
                    out = "Analysis_" + str(analysis_obj.Type) + ".txt"
                analysis_obj.f = open(out, "w")

                if analysis_conf.find('Direction') is not None:
                    analysis_obj.Direction = analysis_conf.find('Direction').text
                if analysis_conf.find('ConstellationID') is not None:
                    analysis_obj.ConstellationID = int(analysis_conf.find('ConstellationID').text)
                if analysis_conf.find('SatelliteID') is not None:
                    analysis_obj.SatelliteID = int(analysis_conf.find('SatelliteID').text)
                if analysis_conf.find('LatitudeRequested') is not None:
                    analysis_obj.LatitudeRequested = float(analysis_conf.find('LatitudeRequested').text)
                if analysis_conf.find('LongitudeRequested'):
                    analysis_obj.LongitudeRequested = float(analysis_conf.find('LongitudeRequested').text)
                if analysis_conf.find('ElevationMask') is not None:
                    analysis_obj.ElevationMask = float(analysis_conf.find('ElevationMask').text)
                if analysis_conf.find('RequiredNumberSatellites') is not None:
                    analysis_obj.RequiredNumberSatellites = int(analysis_conf.find('RequiredNumberSatellites').text)
                if analysis_conf.find('ConstellationName') is not None:
                    analysis_obj.ConstellationName = analysis_conf.find('ConstellationName').text

                self.AnalysisList.append(analysis_obj)


        #self.NumSat = Ground2SpaceLink.NumSat # don't remember why this...???

        self.NumEpoch = floor(86400 * (self.StopDateTime - self.StartDateTime) / self.TimeStep) + 1



