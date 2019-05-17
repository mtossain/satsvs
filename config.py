import xml.etree.ElementTree as ET
from math import floor
from astropy.time import Time

import misc_fn
from constants import *
from analysis import *
from segments import Constellation, Satellite, Station, User, Ground2SpaceLink, User2SpaceLink, Space2SpaceLink
import logging_svs as ls


class AppConfig:
    
    def __init__(self, file_name = None):

        self.constellations = []
        self.satellites = []
        self.stations = []
        self.users = []
        self.gr2sp = []
        self.usr2sp = []
        self.sp2sp = []

        self.analysis = None
        self.file_name = file_name

        self.StartDateTime = 0  # MJD start time simulation
        self.StopDateTime = 0  # MJD stop time simulation
        self.TimeStep = 0  # s
        self.NumTimeSteps = 0
        self.NumEpoch = 0
        self.cnt_epoch = 0
        self.time_gmst = 0  # Loop time
        self.time_mjd = 0  # Loop time
        self.time_str = ''  # Loop time

        self.NumConstellation = 0
        self.NumSat = 0
        self.NumGroundStation = 0
        self.NumUser = 0
        self.NumSpace2Space = 0
        self.NumGround2Space = 0

    def load_satellites(self):

        # Get the list of constellations
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        for constellation in root.iter('Constellation'):
            const = Constellation()
            const.ConstellationID = int(constellation.find('ConstellationID').text)
            const.NumOfSatellites = int(constellation.find('NumOfSatellites').text)
            const.NumberOfPlanes = int(constellation.find('NumOfPlanes').text)
            const.ConstellationName = constellation.find('ConstellationName').text
            ls.logger.info(const.__dict__)
            self.constellations.append(const)
            for satellite in constellation.iter('Satellite'):
                sat = Satellite()
                sat.SatelliteID = int(satellite.find('SatelliteID').text)
                sat.Plane = int(satellite.find('Plane').text)
                sat.ConstellationID = const.ConstellationID
                sat.ConstellationName = const.ConstellationName
                sat.Kepler.EpochMJD = float(satellite.find('EpochMJD').text)
                sat.Kepler.SemiMajorAxis = float(satellite.find('SemiMajorAxis').text)
                sat.Kepler.Eccentricity = float(satellite.find('Eccentricity').text)
                sat.Kepler.Inclination = float(satellite.find('Inclination').text) / 180 * pi
                sat.Kepler.RAAN = float(satellite.find('RAAN').text) / 180 * pi
                sat.Kepler.ArgOfPerigee = float(satellite.find('ArgOfPerigee').text) / 180 * pi
                sat.Kepler.MeanAnomaly = float(satellite.find('MeanAnomaly').text) / 180 * pi
                ls.logger.info(sat.__dict__)
                self.satellites.append(sat)
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
                            ls.logger.info(['Found satellite:', str(sat.SatelliteID),'Kepler Elements', str(sat.Kepler.__dict__)])
                            self.satellites.append(sat)
                            cnt=-1
                        cnt += 1
        self.NumSat = len(self.satellites)
        self.NumConstellation = len(self.constellations)

    def load_stations(self):
        # Get the list of constellations
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        for groundstation in root.iter('GroundStation'):
            gs = Station()
            gs.ConstellationID = int(groundstation.find('ConstellationID').text)
            gs.GroundStationID = int(groundstation.find('GroundStationID').text)
            gs.GroundStationName = groundstation.find('GroundStationName').text
            gs.LLA[0] = float(groundstation.find('Latitude').text) / 180 * pi
            gs.LLA[1] = float(groundstation.find('Longitude').text) / 180 * pi
            gs.LLA[2] = float(groundstation.find('Height').text)
            gs.ReceiverConstellation = groundstation.find('ReceiverConstellation').text
            gs.IdxSatInView = self.NumSat*[999999]

            MaskValues = groundstation.find('ElevationMask').text.split(',')
            for MaskValue in MaskValues:
                gs.ElevationMask.append(float(MaskValue) / 180 * pi)
            if groundstation.find('ElevationMaskMaximum') is not None: # Optional
                MaskMaxValues = groundstation.find('ElevationMaskMaximum').text.split(',')
                for MaskMaxValue in MaskMaxValues:
                    gs.ElevationMaskMaximum.append(MaskMaxValue / 180 * pi)
            else:  # If no maximum set to 90 degrees
                for MaskValue in MaskValues:
                    gs.ElevationMaskMaximum.append(90.0 / 180.0 * pi)

            gs.DeterminePosVelECF()
            ls.logger.info(gs.__dict__)
            self.stations.append(gs)

        self.NumGroundStation = len(self.stations)


    def load_users(self):
        # Get the list of constellations
        tree = ET.parse(self.file_name)
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
                user.IdxSatInView = self.NumSat*[999999]
                MaskValues = userelement.find('ElevationMask').text.split(',')
                user.ElevationMask = list(map(lambda x: float(x) / 180 * pi, MaskValues))
                if userelement.find('ElevationMaskMaximum') is not None:
                    user.ElevationMaskMaximum = list(map(lambda x: float(x) / 180 * pi,
                                                         (userelement.find('ElevationMaskMaximum').text.split(','))))
                else:
                    user.ElevationMaskMaximum = len(MaskValues) * [90.0 / 180.0 * pi]
                user.DeterminePosVelECF()
                self.users.append(user)

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
                        user.LLA[0] = Latitude / 180 * pi
                        user.LLA[1] = Longitude / 180 * pi
                        user.LLA[2] = Height
                        user.NumLat = NumLat
                        user.NumLon = NumLon
                        user.ReceiverConstellation = ReceiverConstellation
                        user.ElevationMask = list(map(lambda x: float(x) / 180 * pi, MaskValues))
                        if userelement.find('ElevationMaskMaximum') is not None:
                            user.ElevationMaskMaximum = list(map(lambda x: float(x) / 180 * pi, MaskMaximumumValues))
                        else:
                            user.ElevationMaskMaximum = len(MaskValues)*[90.0 / 180.0 * pi]

                        user.DeterminePosVelECF()  # Do it once to initialise
                        user.IdxSatInView = self.NumSat*[999999]
                        self.users.append(user)
                        CntUsers += 1

            if userelement.find('Type').text == 'Spacecraft':  # TODO Multiple spacecraft users
                user = User()
                user.UserID = 0
                user.Type = 'Spacecraft'
                user.ReceiverConstellation = userelement.find('ReceiverConstellation').text
                user.IdxSatInView = self.NumSat*[999999]
                MaskValues = userelement.find('ElevationMask').text.split(',')
                user.ElevationMask = list(map(lambda x: float(x) / 180 * pi, MaskValues))
                if userelement.find('ElevationMaskMaximum') is not None:
                    user.ElevationMaskMaximum = list(map(lambda x: float(x) / 180 * pi,
                                                         (userelement.find('ElevationMaskMaximum').text.split(','))))
                else:
                    user.ElevationMaskMaximum = len(MaskValues) * [90.0 / 180.0 * pi]

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
                            ls.logger.info(['Found user satellite:', str(user.UserID), 'Kepler Elements', str(user.Kepler.__dict__)])
                            cnt = -1
                        cnt += 1
                GMSTRequested = misc_fn.MJD2GMST(user.Kepler.EpochMJD)
                user.DeterminePosVelTLE(GMSTRequested, user.Kepler.EpochMJD)
                user.DeterminePosVelECF()
                self.users.append(user)
        ls.logger.info(user.__dict__)
        self.NumUser = len(self.users)

    def setup_links(self):

        cnt_space_link = 0
        cnt_grnd_link = 0
        cnt_usr_link = 0

        Ground2SpaceLink.NumSat = self.NumSat
        for i in range(self.NumGroundStation):
            for j in range(self.NumSat):
                Gr2Sp = Ground2SpaceLink()  # important to have new instance every time, python does by reference!!!
                self.gr2sp.append(Gr2Sp)
                if self.stations[i].ReceiverConstellation[self.satellites[j].ConstellationID - 1] == '1':
                    self.gr2sp[cnt_grnd_link].LinkInUse = True
                else:
                    self.gr2sp[cnt_grnd_link].LinkInUse = False
                cnt_grnd_link += 1
        ls.logger.info(["Loaded: ", cnt_grnd_link, " number of Ground Station to Spacecraft links"])

        User2SpaceLink.NumSat = self.NumSat
        for i in range(self.NumUser):
            for j in range(self.NumSat):
                Usr2Sp = User2SpaceLink()
                self.usr2sp.append(Usr2Sp)  # important to have new instance every time, by reference!!!
                if self.users[i].ReceiverConstellation[self.satellites[j].ConstellationID - 1] == '1':
                    self.usr2sp[cnt_usr_link].LinkInUse = True
                else:
                    self.usr2sp[cnt_usr_link].LinkInUse = False
                cnt_usr_link += 1
        ls.logger.info(["Loaded: ", cnt_usr_link, " number of User to Spacecraft links"])

        for i in range(self.NumSat):
            for j in range(self.NumSat):
                if i != j:  # avoid link to itself
                    Sp2Sp = Space2SpaceLink()  # important to have new instance every time, by reference!!!
                    Sp2Sp.LinkInUse = True  # TODO Receiver Constellation for Sp2Sp
                    Sp2Sp.IdxSat1 = i
                    Sp2Sp.IdxSat2 = j
                    Sp2Sp.IdxSatTransm = self.satellites[i]
                    Sp2Sp.IdxSatRecv = self.satellites[j]
                    self.sp2sp.append(Sp2Sp)
                    cnt_space_link += 1
        ls.logger.info(["Loaded: ", cnt_space_link, " number of Spacecraft to Spacecraft links"])

        # Allocate the memory for the lists of links
        for j in range(self.NumSat):
            self.satellites[j].IdxStationInView = self.NumGroundStation * [999999]
        for j in range(self.NumUser):
            self.users[j].IdxSatInView = self.NumSat*[999999]

    def LoadPropagation(self):
        pass

    def load_simulation(self):
        # Load the simulation parameters
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        for sim in root.iter('SimulationManager'):
            self.StartDateTime = Time(sim.find('StartDate').text,scale='utc').mjd
            self.time_mjd = self.StartDateTime  # initiate the time loop
            self.StopDateTime = Time(sim.find('StopDate').text, scale='utc').mjd
            self.TimeStep = int(sim.find('TimeStep').text)
            ls.logger.info(['Loaded simulation, start:', str(self.StartDateTime), 'stop:', str(self.StopDateTime),
                            'TimeStep', str(self.TimeStep)])

            for analysis_node in root.iter('Analysis'):  # Only one analysis can be performed at a time
                if analysis_node.find('Type').text == 'cov_ground_track':
                    self.analysis = AnalysisCovGroundTrack()
                if analysis_node.find('Type').text == 'cov_depth_of_coverage':
                    self.analysis = AnalysisCovDepthOfCoverage()
                if analysis_node.find('Type').text == 'cov_pass_time':
                    self.analysis = AnalysisCovPassTime()
                if analysis_node.find('Type').text == 'cov_satellite_contour':
                    self.analysis = AnalysisCovSatelliteContour()
                if analysis_node.find('Type').text == 'cov_satellite_highest':
                    self.analysis = AnalysisCovSatelliteHighest()
                if analysis_node.find('Type').text == 'cov_satellite_pvt':
                    self.analysis = AnalysisCovSatellitePvt()
                if analysis_node.find('Type').text == 'cov_sky_angles':
                    self.analysis = AnalysisCovSatelliteSkyAngles()
                if analysis_node.find('Type').text == 'cov_satellite_visible':
                    self.analysis = AnalysisCovSatelliteVisible()
                if analysis_node.find('Type').text == 'cov_satellite_visible_grid':
                    self.analysis = AnalysisCovSatelliteVisibleGrid()
                if analysis_node.find('Type').text == 'cov_satellite_visible_id':
                    self.analysis = AnalysisCovSatelliteVisibleId()
                self.analysis.read_config(analysis_node)  # Read the configuration for the specific analysis

        self.NumEpoch = floor(86400 * (self.StopDateTime - self.StartDateTime) / self.TimeStep)



