import xml.etree.ElementTree as ET
from math import floor, radians, degrees
from astropy.time import Time

from constants import *
from analysis import *
from segments import Constellation, Satellite, Station, User, Ground2SpaceLink, User2SpaceLink, Space2SpaceLink
import logging_svs as ls


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

class AppConfig:
    
    def __init__(self, file_name=None):

        self.constellations = []
        self.satellites = []
        self.stations = []
        self.users = []
        self.gr2sp = []
        self.usr2sp = []
        self.sp2sp = []

        self.analysis = None
        self.file_name = file_name

        self.start_time = 0  # MJD start time simulation
        self.stop_time = 0  # MJD stop time simulation
        self.time_step = 0  # in seconds
        self.num_epoch = 0  # Number of epochs in simulation
        self.cnt_epoch = 0
        self.time_gmst = 0  # Loop time
        self.time_mjd = 0  # Loop time
        self.time_str = ''  # Loop time

        self.include_gr2sp = True
        self.include_usr2sp = True
        self.include_sp2sp = True

        self.num_constellation = 0
        self.num_sat = 0
        self.num_station = 0
        self.num_user = 0
        self.num_sp2sp = 0
        self.num_st2sp = 0
        self.num_usr2sp = 0

    def load_satellites(self):

        # Get the list of constellations
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        for constellation in root.iter('Constellation'):
            const = Constellation()
            const.constellation_id = int(constellation.find('ConstellationID').text)
            const.num_sat = int(constellation.find('NumOfSatellites').text)
            const.num_planes = int(constellation.find('NumOfPlanes').text)
            const.constellation_name = constellation.find('ConstellationName').text
            ls.logger.info(const.__dict__)
            self.constellations.append(const)
            for satellite in constellation.iter('Satellite'):
                sat = Satellite()
                sat.sat_id = int(satellite.find('SatelliteID').text)
                sat.plane = int(satellite.find('Plane').text)
                sat.constellation_id = const.constellation_id
                sat.constellation_name = const.constellation_name
                sat.kepler.epoch_mjd = float(satellite.find('EpochMJD').text)
                sat.kepler.semi_major_axis = float(satellite.find('SemiMajorAxis').text)
                sat.kepler.eccentricity = float(satellite.find('Eccentricity').text)
                sat.kepler.inclination = radians(float(satellite.find('Inclination').text))
                sat.kepler.right_ascension = radians(float(satellite.find('RAAN').text))
                sat.kepler.arg_perigee = radians(float(satellite.find('ArgOfPerigee').text))
                sat.kepler.mean_anomaly = radians(float(satellite.find('MeanAnomaly').text))
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
                            sat.sat_id = int(line[2:7])  # NORAD Catalog Number
                            sat.constellation_id = const.constellation_id
                            sat.constellation_name = const.constellation_name
                            full_year = "20" + line[18:20]
                            epoch_yr = int(full_year)
                            epoch_doy = float(line[20:31])
                            sat.kepler.epoch_mjd = misc_fn.yyyy_doy2mjd(epoch_yr, epoch_doy)
                        if cnt == 2:
                            mu = GM_EARTH
                            mean_motion = float(line[52:59]) / 86400 * 2 * math.PI  # rad / s
                            sat.kepler.semi_major_axis = pow((mu / (pow(mean_motion, 2))), (1.0 / 3.0))
                            eccentricity = "0." + line[26:33]
                            sat.kepler.eccentricity = float(eccentricity)
                            sat.kepler.inclination = radians(float(line[8:15]))
                            sat.kepler.right_ascension = radians(float(line[17:24]))
                            sat.kepler.arg_perigee = radians(float(line[34:41]) )
                            sat.kepler.mean_anomaly = radians(float(line[43:50]))
                            ls.logger.info(['Found satellite:', str(sat.sat_id),'Kepler Elements', str(sat.kepler.__dict__)])
                            self.satellites.append(sat)
                            cnt=-1
                        cnt += 1
        self.num_sat = len(self.satellites)
        self.num_constellation = len(self.constellations)

    def load_stations(self):
        # Get the list of constellations
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        for station in root.iter('GroundStation'):
            gs = Station()
            gs.constellation_id = int(station.find('ConstellationID').text)
            gs.station_id = int(station.find('GroundStationID').text)
            gs.station_name = station.find('GroundStationName').text
            gs.lla[0] = radians(float(station.find('Latitude').text))
            gs.lla[1] = radians(float(station.find('Longitude').text))
            gs.lla[2] = float(station.find('Height').text)
            gs.rx_constellation = station.find('ReceiverConstellation').text
            gs.idx_sat_in_view = self.num_sat*[999999]

            mask_values = station.find('ElevationMask').text.split(',')
            for mask_value in mask_values:
                gs.elevation_mask.append(radians(float(mask_value)))
            if station.find('ElevationMaskMaximum') is not None: # Optional
                mask_max_values = station.find('ElevationMaskMaximum').text.split(',')
                for mask_max_value in mask_max_values:
                    gs.el_mask_max.append(radians(mask_max_value))
            else:  # If no maximum set to 90 degrees
                for mask_value in mask_values:
                    gs.el_mask_max.append(radians(90.0))

            gs.det_pvt_ecf()
            ls.logger.info(gs.__dict__)
            self.stations.append(gs)

        self.num_station = len(self.stations)

    def load_users(self):
        # Get the list of constellations
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        for user_element in root.iter('UserSegment'):

            if user_element.find('Type').text == 'Static':
                user = User()
                user.type = user_element.find('Type').text
                user.user_id = 0
                user.lla[0] = radians(float(user_element.find('Latitude').text))
                user.lla[1] = radians(float(user_element.find('Longitude').text))
                user.lla[2] = float(user_element.find('Height').text)
                user.rx_constellation = user_element.find('ReceiverConstellation').text
                user.idx_sat_in_view = self.num_sat*[999999]
                mask_values = user_element.find('ElevationMask').text.split(',')
                user.elevation_mask = [radians(float(n)) for n in mask_values]
                if user_element.find('ElevationMaskMaximum') is not None:
                    mask_max_values = user_element.find('ElevationMaskMaximum').text.split(',')
                    user.el_mask_max = [radians(float(n))for n in mask_max_values]
                else:
                    user.el_mask_max = len(mask_values) * [radians(90.0)]
                user.det_pvt_ecf()
                self.users.append(user)

            if user_element.find('Type').text == 'Grid':
                lat_min = float(user_element.find('LatMin').text)
                lat_max = float(user_element.find('LatMax').text)
                lon_min = float(user_element.find('LonMin').text)
                lon_max = float(user_element.find('LonMax').text)
                lat_step = float(user_element.find('LatStep').text)
                lon_step = float(user_element.find('LonStep').text)
                num_lat = int((lat_max - lat_min) / lat_step) + 1
                num_lon = int((lon_max - lon_min) / lon_step) + 1
                height = float(user_element.find('Height').text)
                rx_constellation = user_element.find('ReceiverConstellation').text
                mask_values = user_element.find('ElevationMask').text.split(',')
                mask_max_values = []
                if user_element.find('ElevationMaskMaximum') is not None:
                    mask_max_values = user_element.find('ElevationMaskMaximum').text.split(',')

                # Now make the full list of users
                cnt_users = 0
                for i in range(num_lat):
                    latitude = lat_min + lat_step * i
                    for j in range(num_lon):
                        longitude = lon_min + lon_step * j
                        user = User()
                        user.type = "Grid"
                        user.user_id = cnt_users
                        user.UserName = "Grid"
                        user.lla[0] = radians(latitude)
                        user.lla[1] = radians(longitude)
                        user.lla[2] = height
                        user.num_lat = num_lat
                        user.num_lon = num_lon
                        user.rx_constellation = rx_constellation
                        user.elevation_mask = [radians(float(n)) for n in mask_values]
                        if user_element.find('ElevationMaskMaximum') is not None:
                            user.el_mask_max = [radians(float(n)) for n in mask_max_values]
                        else:
                            user.el_mask_max = len(mask_values) * [radians(90.0)]
                        user.det_pvt_ecf()  # Do it once to initialise
                        user.idx_sat_in_view = self.num_sat*[999999]
                        self.users.append(user)
                        cnt_users += 1

            if user_element.find('Type').text == 'Spacecraft':  # TODO Multiple spacecraft users
                user = User()
                user.user_id = 0
                user.type = 'Spacecraft'
                user.rx_constellation = user_element.find('ReceiverConstellation').text
                user.idx_sat_in_view = self.num_sat*[999999]
                mask_values = user_element.find('ElevationMask').text.split(',')
                user.elevation_mask = [radians(float(n)) for n in mask_values]
                if user_element.find('ElevationMaskMaximum') is not None:
                    mask_max_values = user_element.find('ElevationMaskMaximum').text.split(',')
                    user.el_mask_max = [radians(float(n)) for n in mask_max_values]
                else:
                    user.el_mask_max = len(mask_values) * [radians(90.0)]

                user.tle_file_name = user_element.find('TLEFileName').text
                with open(user.tle_file_name) as file:
                    cnt = 0
                    for line in file:
                        if cnt == 0:
                            user.UserName = line
                        if cnt == 1:
                            user.user_id = int(line[2:7])  # NORAD Catalog Number
                            full_year = "20" + line[18:20]
                            epoch_yr = int(full_year)
                            epoch_doy = float(line[20:31])
                            user.kepler.epoch_mjd = misc_fn.yyyy_doy2mjd(epoch_yr, epoch_doy)
                        if cnt == 2:
                            mu = GM_EARTH
                            mean_motion = float(line[52:59]) / 86400 * 2 * math.PI  # rad / s
                            user.kepler.semi_major_axis = pow((mu / (pow(mean_motion, 2))), (1.0 / 3.0))
                            eccentricity = "0." + line[26:33]
                            user.kepler.eccentricity = float(eccentricity)
                            user.kepler.inclination = radians(float(line[8:15]))
                            user.kepler.right_ascension = radians(float(line[17:24]))
                            user.kepler.arg_perigee = radians(float(line[34:41]))
                            user.kepler.mean_anomaly = radians(float(line[43:50]))
                            ls.logger.info(['Found user satellite:', str(user.user_id), 'Kepler Elements', str(user.kepler.__dict__)])
                            cnt = -1
                        cnt += 1
                gmst_requested = misc_fn.mjd2gmst(user.kepler.epoch_mjd)
                user.det_pvt_tle(gmst_requested, user.kepler.epoch_mjd)
                user.det_pvt_ecf()
                self.users.append(user)
        ls.logger.info(user.__dict__)
        self.num_user = len(self.users)

    def setup_links(self):

        # Setup the list of objects properly indexed in 2d, eg. [0][0]
        # It is important to have new instance every time, python does by reference!!!
        # STATION <->SATELLITE LINKS
        self.gr2sp = [[Ground2SpaceLink() for j in range(self.num_sat)] for i in range(self.num_station)]
        for idx_station in range(self.num_station):
            for idx_sat in range(self.num_sat):
                if self.stations[idx_station].rx_constellation[self.satellites[idx_sat].constellation_id - 1] == '1':
                    self.gr2sp[idx_station][idx_sat].link_in_use = True
                else:
                    self.gr2sp[idx_station][idx_sat].link_in_use = False
        ls.logger.info(["Loaded: ", len(self.gr2sp), " number of Ground Station to Spacecraft links"])

        # USER <->SATELLITE LINKS
        self.usr2sp = [[User2SpaceLink() for j in range(self.num_sat)] for i in range(self.num_user)]
        for idx_user in range(self.num_user):
            for j in range(self.num_sat):
                if self.users[idx_user].rx_constellation[self.satellites[idx_sat].constellation_id - 1] == '1':
                    self.usr2sp[idx_user][idx_sat].link_in_use = True
                else:
                    self.usr2sp[idx_user][idx_sat].link_in_use = False
        ls.logger.info(["Loaded: ", len(self.usr2sp), " number of User to Spacecraft links"])

        # SATELLITE <->SATELLITE LINKS
        self.sp2sp = [[Space2SpaceLink() for j in range(self.num_sat)] for i in range(self.num_sat)]
        for idx_sat1 in range(self.num_sat):   # TODO Receiver Constellation for Sp2Sp
            for idx_sat2 in range(self.num_sat):
                if idx_sat1 != idx_sat2:  # avoid link to itself
                    self.sp2sp[idx_sat1][idx_sat2].idx_sat_tx = idx_sat1
                    self.sp2sp[idx_sat1][idx_sat2].idx_sat_rx = idx_sat2
        ls.logger.info(["Loaded: ", len(self.sp2sp), " number of Spacecraft to Spacecraft links"])

        # Allocate the memory for the lists of links
        for j in range(self.num_sat):
            self.satellites[j].idx_stat_in_view = self.num_station * [999999]
        for j in range(self.num_user):
            self.users[j].idx_sat_in_view = self.num_sat*[999999]

    def load_simulation(self):
        # Load the simulation parameters
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        for sim in root.iter('SimulationManager'):
            self.start_time = Time(sim.find('StartDate').text,scale='utc').mjd
            self.time_mjd = self.start_time  # initiate the time loop
            self.stop_time = Time(sim.find('StopDate').text, scale='utc').mjd
            self.time_step = int(sim.find('TimeStep').text)
            self.include_gr2sp = str2bool(sim.find('IncludeStation2SpaceLinks').text)
            self.include_usr2sp = str2bool(sim.find('IncludeUser2SpaceLinks').text)
            self.include_sp2sp = str2bool(sim.find('IncludeSpace2SpaceLinks').text)
            ls.logger.info(['Loaded simulation, start:', str(self.start_time), 'stop:', str(self.stop_time),
                            'TimeStep', str(self.time_step)])

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

        self.num_epoch = floor(86400 * (self.stop_time - self.start_time) / self.time_step)
