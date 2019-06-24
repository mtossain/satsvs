# Standard python packages
import numpy as np
from math import floor, degrees
from astropy.coordinates import EarthLocation
from astropy.time import Time
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
# Modules from the project
from src.constants import PI, R_EARTH, OMEGA_EARTH
from src import misc_fn


class KeplerSet:

    def __init__(self):
        self.epoch_mjd = 0.0  # Epoch of orbit in MJD
        self.semi_major_axis = 0.0
        self.eccentricity = 0.0
        self.inclination = 0.0
        self.right_ascension = 0.0  # Right ascension ascending node
        self.arg_perigee = 0.0  # Argument of perigee
        self.mean_anomaly = 0.0


class Constellation:

    def __init__(self):
        self.constellation_id = 0
        self.constellation_name = ''
        self.num_sat = 0
        self.num_planes = 0
        self.rx_constellation = ''
        self.elevation_mask = []  # Could be varying over azimuth...
        self.el_mask_max = []  # Could be varying over azimuth...
        self.uere_list = []  # Varies over elevation

        self.tle_file_name = None
        self.obs_inci_angle_start = None
        self.obs_inci_angle_stop = None
        self.obs_swath_start = None
        self.obs_swath_stop = None
        self.frontal_area = None
        self.mass = None


class Satellite:

    def __init__(self):

        self.constellation_id = 0  # Refers to a constellation
        self.constellation_name = ''  # Refers to a constellation
        self.sat_id = 0  # Satellite ID
        self.plane = 0  # Plane the satellite is in
        self.name = ''  # Name of the satellite
        self.rx_constellation = ''  # Which constellations can this satellite receive (for SP2SP)

        self.elevation_mask = []  # Could be varying over azimuth...
        self.el_mask_max = []  # Could be varying over azimuth...
        self.uere_list = []  # Varies over elevation

        self.kepler = KeplerSet()  # Containing Kepler set for orbit
        self.tle_line1 = ''  # If TLE file, then contains the TLE first line
        self.tle_line2 = ''  # If TLE file, then contains the TLE second line
        self.frontal_area = None
        self.mass = None
        self.satrec = None # Object needed for SGP4 propagation

        self.pos_eci = np.zeros(3)  # Satellite position in ECI
        self.vel_eci = np.zeros(3)  # Satellite velocity in ECI
        self.pos_ecf = np.zeros(3)  # Satellite position in ECF
        self.vel_ecf = np.zeros(3)  # Satellite velocity in ECF
        self.lla = np.zeros(3)  # For ground track

        self.idx_stat_in_view = []  # Indices of station which are in view
        self.idx_sat_in_view = []  # Indices of other satellites which are in view

        self.metric = []  # For analysis purposes

        self.obs_inci_angle_start = None  # If satellite contains instrument
        self.obs_inci_angle_stop = None
        self.obs_swath_start = None  # Alternatively swath width instrument
        self.obs_swath_stop = None

        self.p1 = np.zeros(3) # four corners of the swath
        self.p2 = np.zeros(3)  # four corners of the swath
        self.p3 = np.zeros(3)  # four corners of the swath
        self.p4 = np.zeros(3)  # four corners of the swath

    def det_posvel_eci_keplerian(self, mjd_requested):
        self.pos_eci, self.vel_eci = misc_fn.kep2xyz(mjd_requested, self.kepler.epoch_mjd,
                                                     self.kepler.semi_major_axis, self.kepler.eccentricity,
                                                     self.kepler.inclination, self.kepler.right_ascension,
                                                     self.kepler.arg_perigee, self.kepler.mean_anomaly)

    def det_posvel_eci_sgp4(self, time_req):
        self.pos_eci,self.vel_eci = self.satrec.propagate(time_req.year, time_req.month, time_req.day,
                                      time_req.hour, time_req.minute, time_req.second)
        self.pos_eci = np.array(self.pos_eci)*1000  # sgp4 outputs in km
        self.vel_eci = np.array(self.vel_eci)*1000  # sgp4 outputs in km/s

    def det_posvel_ecf(self, gmst_requested):  # ECF
        self.pos_ecf, self.vel_ecf = misc_fn.spin_vector(-gmst_requested, self.pos_eci, self.vel_eci)  # assume ECI and GMST computed elsewhere

    def det_lla(self):
        self.lla = misc_fn.xyz2lla(self.pos_ecf)


class Station:

    def __init__(self):
        self.constellation_id = 0
        self.station_id = 0
        self.station_name = ''
        self.rx_constellation = ''
        self.elevation_mask = []  # Could be varying over azimuth...
        self.el_mask_max = []  # Could be varying over azimuth...

        self.pos_eci = np.zeros(3)
        self.vel_eci = np.zeros(3)
        self.pos_ecf = np.zeros(3)
        self.vel_ecf = np.zeros(3)
        self.lla = np.zeros(3)

        self.idx_sat_in_view = []  # Indices of satellites which are in view

    def det_posvel_ecf(self):
        self.pos_ecf = misc_fn.lla2xyz(self.lla)  # ECF

    def det_posvel_eci(self, gmst_requested):
        self.pos_eci, self.vel_eci = misc_fn.spin_vector(gmst_requested, self.pos_ecf, self.vel_ecf)  # Requires the ECF to be computed first
        self.vel_eci = np.cross(self.pos_eci, [0, 0, -OMEGA_EARTH])

    def det_posvel_eci_astropy(self):
        # Same as det_posvel_eci but now using Astropy
        # It is however much slower: about .05s vs 0.000005s
        location = EarthLocation.from_geodetic(degrees(self.lla[1]), degrees(self.lla[0]), self.lla[2])
        mjd = Time(55324.0, format='mjd')
        eci = location.get_gcrs_posvel(mjd)
        self.pos_eci = eci[0].xyz.value
        self.vel_eci = eci[1].xyz.value


class User:

    def __init__(self):

        self.type = ''  # Static, grid or spacecraft
        self.user_id = 0
        self.rx_constellation = ''
        self.elevation_mask = []  # Could be varying over azimuth...
        self.el_mask_max = []  # Could be varying over azimuth...

        self.pos_eci = np.zeros(3)
        self.vel_eci = np.zeros(3)
        self.pos_ecf = np.zeros(3)
        self.vel_ecf = np.zeros(3)
        self.norm_ecf = 0  # For speeding up processes
        self.lla = np.zeros(3)
        self.num_lat = 0
        self.num_lon = 0

        self.idx_sat_in_view = []  # Indices of satellites which are in view

        self.tle_file_name = ''  # In case user is a spacecraft
        self.kepler = KeplerSet()  # In case user is a spacecraft

        self.metric = []  # For analysis purposes

    def det_posvel_ecf(self):
        self.pos_ecf = misc_fn.lla2xyz(self.lla)

    def det_posvel_eci(self, gmst_requested):
        self.pos_eci, self.vel_eci = misc_fn.spin_vector(gmst_requested, self.pos_ecf, self.vel_ecf)  # Requires the ECF to be computed first
        self.vel_eci = np.cross(self.pos_eci, [0, 0, -OMEGA_EARTH])

    def det_posvel_tle(self, gmst_requested, mjd_requested):  # For spacecraft user
        # Compute ECF and ECI coordinates from MJD and TLE set
        self.pos_eci, self.vel_eci = misc_fn.kep2xyz(mjd_requested, self.kepler.epoch_mjd,
                                                     self.kepler.semi_major_axis, self.kepler.eccentricity,
                                                     self.kepler.inclination, self.kepler.right_ascension,
                                                     self.kepler.arg_perigee, self.kepler.mean_anomaly)
        self.pos_ecf, self.vel_ecf = misc_fn.spin_vector(-gmst_requested, self.pos_eci, self.vel_eci)  # ECF


class Ground2SpaceLink:

    def __init__(self):

        self.link_in_use = True  # Defines whether ground asset is using this satellite (from ReceiverConstellation)

        self.azimuth = 0.0  # Radians
        self.elevation = 0.0  # Radians
        self.azimuth2 = 0.0  # Receiver as seen from satellite
        self.elevation2 = 0.0  # Receiver as seen from satellite (equivalent to off-nadir from satellite)

        self.gr2sp_ecf = np.zeros(3)  # m
        self.distance = 0.0  # m

        self.metric = []  # For analysis purposes

    def compute_link(self, station, satellite):
        # For reasons of speed this function is coded inline, since this function is done for every time step,
        # for every combination of user and satellite. Also the computation of azimuth is not done standard,
        # since it takes more time and not needed by all analysis

        self.gr2sp_ecf = satellite.pos_ecf - station.pos_ecf

        self.distance = np.linalg.norm(self.gr2sp_ecf)

        self.azimuth, self.elevation = misc_fn.calc_az_el(satellite.pos_ecf, station.pos_ecf)  # From Station to Satellite
        self.azimuth2, self.elevation2 = misc_fn.calc_az_el(station.pos_ecf, satellite.pos_ecf)  # From Satellite to Station

    def check_masking(self, station):
        # Check whether satellite is above masking angle defined for station/user
        # Optionally also a maximum mask angle is defined
        in_view = False
        number_of_masks = len(station.elevation_mask)

        if number_of_masks == 1:
            if station.elevation_mask[0] < self.elevation < station.el_mask_max[0]:
                in_view = True
        else:  # More than one mask
            az_cake_piece_angle = 2 * PI / number_of_masks
            idx_az = int(floor(self.azimuth / az_cake_piece_angle))
            if station.elevation_mask[idx_az] < self.elevation < station.el_mask_max[idx_az]:
                in_view = True

        return in_view


class User2SpaceLink:

    def __init__(self):

        self.link_in_use = True  # Defines whether ground asset is using this satellite (from ReceiverConstellation)

        self.azimuth = 0.0  # Radians
        self.elevation = 0.0  # Radians
        self.azimuth2 = 0.0  # Receiver as seen from satellite
        self.elevation2 = 0.0  # Receiver as seen from satellite (equivalent to off-nadir from satellite)

        self.usr2sp_ecf = np.zeros(3)  # m
        self.distance = 0.0  # m

        self.metric = []  # For analysis purposes

    def compute_link(self, user, satellite):

        self.usr2sp_ecf = satellite.pos_ecf - user.pos_ecf

        self.distance = np.linalg.norm(self.usr2sp_ecf)

        self.azimuth, self.elevation = misc_fn.calc_az_el(satellite.pos_ecf, user.pos_ecf)

    def check_masking(self, user):
        # Check whether satellite is above masking angle defined for station/user
        # Optionally also a maximum mask angle is defined

        in_view = False
        number_of_masks = len(user.elevation_mask)

        if number_of_masks == 1:
            if user.elevation_mask[0] < self.elevation < user.el_mask_max[0]:
                in_view = True
        else:  # More than one mask
            az_cake_piece_angle = 2 * PI / number_of_masks
            idx_az = int(floor(self.azimuth / az_cake_piece_angle))
            if user.elevation_mask[idx_az] < self.elevation < user.el_mask_max[idx_az]:
                in_view = True

        return in_view


class Space2SpaceLink:

    def __init__(self):

        self.link_in_use = True  # Defines whether space asset is using this satellite (from ReceiverConstellation)

        self.idx_sat_tx = 0  # Pointer to transmitting sat
        self.idx_sat_rx = 0  # Pointer to receiving sat

        self.azimuth = 0.0  # Radians
        self.elevation = 0.0  # Radians (is really the LOS-NADIR angle from one sat to the other)
        self.azimuth2 = 0.0  # Radians
        self.elevation2 = 0.0  # Radians

        self.sp2sp_ecf = np.zeros(3)
        self.distance = 0  # m

        self.metric = []  # For analysis purposes

    def compute_link(self, sat_1, sat_2):
        # Compute distance, vector and if Earth is in between...
        # Returns True if the Earth is not in between both satellites

        self.sp2sp_ecf = sat_2.pos_ecf - sat_1.pos_ecf

        self.distance = np.linalg.norm(self.sp2sp_ecf)

        self.azimuth, self.elevation = misc_fn.calc_az_el(sat_1.pos_ecf, sat_2.pos_ecf)
        self.azimuth2, self.elevation2 = misc_fn.calc_az_el(sat_2.pos_ecf, sat_1.pos_ecf)

    def check_masking(self, sat_1, sat_2):

        # Check whether satellite is above masking angle defined for sat/sat
        # Optionally also a maximum mask angle is defined

        in_view_elevation = False

        num_masks = len(sat_1.elevation_mask)  # If only one value
        if num_masks == 0:  # omnidirectional antenna
            in_view_elevation = True
        if num_masks == 1:
            # checking for mask of satellite 1
            if sat_1.elevation_mask[0] < self.elevation < sat_1.el_mask_max[0]:
                in_view_elevation = True
        if num_masks > 1:  # More than one mask
            az_cake_piece_angle = 2 * PI / num_masks
            idx_az = int(floor(self.azimuth / az_cake_piece_angle))
            if sat_1.elevation_mask[idx_az] < self.elevation < sat_1.el_mask_max[idx_az]:
                in_view_elevation = True

        # Also check whether the link is not passing through the Earth
        intersect_earth, i_x1, i_x2 = misc_fn.line_sphere_intersect(sat_1.pos_eci, sat_2.pos_eci, R_EARTH, [0, 0, 0])

        if not intersect_earth and in_view_elevation:
            return True
        else:
            return False

