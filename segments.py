# Standard python packages
import numpy as np
from math import floor, degrees
from astropy.coordinates import EarthLocation
from astropy.time import Time
# Modules from the project
from constants import PI, R_EARTH, OMEGA_EARTH
import misc_fn


class KeplerSet:

    def __init__(self):
        self.epoch_mjd = 0.0
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
        self.tle_file_name = ''
        self.rx_constellation = ''


class Satellite:

    def __init__(self):

        self.constellation_id = 0
        self.constellation_name = ''
        self.sat_id = 0
        self.plane = 0
        self.name = ''
        self.rx_constellation = ''

        self.antenna_mask = []  # Could be varying over azimuth...
        self.antenna_mask_max = []  # Could be varying over azimuth...

        self.kepler = KeplerSet()

        self.pvt_eci = 6*[0.0]
        self.pvt_ecf = 6*[0.0]
        self.lla = 3*[0.0]  # For ground track

        self.idx_stat_in_view = []  # Indices of station which are in view
        self.idx_sat_in_view = []

        self.metric = []  # For analysis purposes

    def det_posvel_eci(self, mjd_requested):
        self.pvt_eci = misc_fn.kep2xyz(mjd_requested, self.kepler)

    def det_posvel_ecf(self, gmst_requested):  # ECF
        self.pvt_ecf = misc_fn.spin_vector(-gmst_requested, self.pvt_eci)  # assume ECI and GMST computed elsewhere

    def det_lla(self):
        self.lla = misc_fn.xyz2lla(self.pvt_ecf)


class Station:

    def __init__(self):
        self.constellation_id = 0
        self.station_id = 0
        self.station_name = ''
        self.rx_constellation = ''
        self.elevation_mask = []  # Could be varying over azimuth...
        self.el_mask_max = []  # Could be varying over azimuth...

        self.pvt_eci = 6*[0.0]
        self.pvt_ecf = 6*[0.0]
        self.lla = 3*[0.0]

        self.idx_sat_in_view = []  # Indices of satellites which are in view

    def det_posvel_ecf(self):
        self.pvt_ecf[0:3] = misc_fn.lla2xyz(self.lla)  # ECF

    def det_posvel_eci(self, gmst_requested):
        self.pvt_eci = misc_fn.spin_vector(gmst_requested, self.pvt_ecf)  # Requires the ECF to be computed first
        self.pvt_eci[3:6] = np.cross(self.pvt_eci[0:3], [0, 0, -OMEGA_EARTH])

    def det_posvel_eci_astropy(self):
        # Same as det_posvel_eci but now using Astropy
        # It is however much slower: about .05s vs 0.000005s
        location = EarthLocation.from_geodetic(degrees(self.lla[1]), degrees(self.lla[0]), self.lla[2])
        mjd = Time(55324.0, format='mjd')
        eci = location.get_gcrs_posvel(mjd)
        self.pvt_eci[0:3] = eci[0].xyz.value
        self.pvt_eci[3:6] = eci[1].xyz.value


class User:

    def __init__(self):

        self.type = ''  # Static, grid or spacecraft
        self.user_id = 0
        self.rx_constellation = ''
        self.elevation_mask = []  # Could be varying over azimuth...
        self.el_mask_max = []  # Could be varying over azimuth...

        self.pvt_eci = 6*[0.0]
        self.pvt_ecf = 6*[0.0]
        self.lla = 3*[0.0]
        self.num_lat = 0
        self.num_lon = 0

        self.idx_sat_in_view = []  # Indices of satellites which are in view

        self.tle_file_name = ''  # In case user is a spacecraft
        self.kepler = KeplerSet()  # In case user is a spacecraft

        self.metric = []  # For analysis purposes

    def det_posvel_ecf(self):
        self.pvt_ecf[0:3] = misc_fn.lla2xyz(self.lla)

    def det_posvel_eci(self, gmst_requested):
        self.pvt_eci = misc_fn.spin_vector(gmst_requested, self.pvt_ecf)  # Requires the ECF to be computed first
        self.pvt_eci[3:6] = np.cross(self.pvt_eci[0:3], [0, 0, -OMEGA_EARTH])

    def det_posvel_tle(self, gmst_requested, mjd_requested):  # For spacecraft user
        # Compute ECF and ECI coordinates from MJD and TLE set
        self.pvt_eci = misc_fn.kep2xyz(mjd_requested, self.kepler)  # ECI
        self.pvt_ecf = misc_fn.spin_vector(-gmst_requested, self.pvt_eci)  # ECF


class Ground2SpaceLink:

    def __init__(self):

        self.link_in_use = True  # Defines whether ground asset is using this satellite (from ReceiverConstellation)

        self.azimuth = 0.0  # Radians
        self.elevation = 0.0  # Radians
        self.azimuth2 = 0.0  # Receiver as seen from satellite
        self.elevation2 = 0.0  # Receiver as seen from satellite (equivalent to off-nadir from satellite)

        self.gr2sp_ecf = [0.0, 0.0, 0.0]  # m
        self.distance = 0.0  # m

        self.metric = []  # For analysis purposes

    def compute_link(self, station, satellite):
        # For reasons of speed this function is coded inline, since this function is done for every time step,
        # for every combination of user and satellite. Also the computation of azimuth is not done standard,
        # since it takes more time and not needed by all analysis

        self.gr2sp_ecf = [satellite.pvt_ecf[i] - station.pvt_ecf[i] for i in range(3)]

        self.distance = np.linalg.norm(self.gr2sp_ecf)

        self.azimuth, self.elevation = misc_fn.calc_az_el(satellite.pvt_ecf, station.pvt_ecf)  # From Station to Satellite
        self.azimuth2, self.elevation2 = misc_fn.calc_az_el(station.pvt_ecf, satellite.pvt_ecf)  # From Satellite to Station

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

        self.usr2sp_ecf = [0.0,0.0,0.0]  # m
        self.distance = 0.0  # m

        self.metric = []  # For analysis purposes

    def compute_link(self, user, satellite):

        self.usr2sp_ecf = [satellite.pvt_ecf[i] - user.pvt_ecf[i] for i in range(3)]

        self.distance = np.linalg.norm(self.usr2sp_ecf)

        self.azimuth, self.elevation = misc_fn.calc_az_el(satellite.pvt_ecf, user.pvt_ecf)

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

        self.link_in_use = True  # Defines whether ground asset is using this satellite (from ReceiverConstellation)

        self.idx_sat_tx = 0  # Pointer to transmitting sat
        self.idx_sat_rx = 0  # Pointer to receiving sat

        self.azimuth = 0.0  # Radians
        self.elevation = 0.0  # Radians (is really the LOS-NADIR angle from one sat to the other)
        self.azimuth2 = 0.0  # Radians
        self.elevation2 = 0.0  # Radians

        self.sp2sp_ecf = 3*[0]
        self.distance = 0  # m

        self.metric = []  # For analysis purposes

    def compute_link(self, sat_1, sat_2):
        # Compute distance, vector and if Earth is in between...
        # Returns True if the Earth is not in between both satellites

        self.sp2sp_ecf = [sat_2.pvt_ecf[i] - sat_1.pvt_ecf[i] for i in range(3)]

        self.distance = np.linalg.norm(self.sp2sp_ecf)

        self.azimuth, self.elevation = misc_fn.calc_az_el(sat_1.pvt_ecf, sat_2.pvt_ecf)
        self.azimuth2, self.elevation2 = misc_fn.calc_az_el(sat_2.pvt_ecf, sat_1.pvt_ecf)

    def check_masking(self, sat_1, sat_2):

        # Check whether satellite is above masking angle defined for sat/sat
        # Optionally also a maximum mask angle is defined

        in_view_elevation = False

        num_masks = len(sat_1.antenna_mask)  # If only one value
        if num_masks == 0:  # omnidirectional antenna
            in_view_elevation = True
        if num_masks == 1:
            # checking for mask of satellite 1
            if sat_1.antenna_mask[0] < self.elevation < sat_1.antenna_mask_max[0]:
                in_view_elevation = True
        if num_masks > 1:  # More than one mask
            az_cake_piece_angle = 2 * PI / num_masks
            idx_az = int(floor(self.azimuth / az_cake_piece_angle))
            if sat_1.antenna_mask[idx_az] < self.elevation < sat_1.antenna_mask_max[idx_az]:
                in_view_elevation = True

        # Also check whether the link is not passing through the Earth
        intersect_earth, i_x1, i_x2 = misc_fn.line_sphere_intersect(sat_1.pvt_eci, sat_2.pvt_eci, R_EARTH, [0,0,0])

        if not intersect_earth and in_view_elevation:
            return True
        else:
            return False

