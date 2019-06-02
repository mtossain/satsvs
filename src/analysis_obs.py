import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
from numpy.linalg import norm
from math import sin, cos, asin, degrees, radians
from numba import jit
# Project modules
from constants import R_EARTH
from analysis import AnalysisBase, AnalysisObs
import misc_fn


class AnalysisObsSwathConical(AnalysisBase, AnalysisObs):

    def __init__(self):
        super().__init__()
        self.ortho_view_latitude = None
        self.revisit = None
        self.statistic = None

    def read_config(self, node):
        if node.find('OrthoViewLatitude') is not None:
            self.ortho_view_latitude = float(node.find('OrthoViewLatitude').text)
        if node.find('Revisit') is not None:
            self.revisit = misc_fn.str2bool(node.find('Revisit').text)
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text.lower()

    def before_loop(self, sm):
        for satellite in sm.satellites:
            idx_found = 0
            for idx, constellation in enumerate(sm.constellations):
                if satellite.constellation_id == constellation.constellation_id:
                    idx_found = idx
            const = sm.constellations[idx_found]
            sat_altitude = satellite.kepler.semi_major_axis - R_EARTH
            if const.obs_swath_stop is not None:  # if swath defined by swath length rather than incidence
                satellite.obs_incl_angle_stop = misc_fn.incl_from_swath(
                    const.obs_swath_stop, R_EARTH, sat_altitude)
            else:
                satellite.obs_incl_angle_stop = const.obs_incl_angle_stop
            alfa_critical = asin(R_EARTH / (R_EARTH + sat_altitude))  # If incidence angle shooting off Earth -> error
            if satellite.obs_incl_angle_stop > alfa_critical:
                ls.logger.error(f'Inclination angle stop: {degrees(satellite.obs_incl_angle_stop)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()

        self.user_pos = np.zeros((len(sm.users),4))
        self.user_metric = np.zeros((len(sm.users), sm.num_epoch), dtype=np.int8)
        for idx_user, user in enumerate(sm.users):
            self.user_pos[idx_user,0:3] = np.array(user.posvel_ecf[0:3])
            self.user_pos[idx_user,3] = norm(self.user_pos[idx_user,0:3])

    def in_loop(self, sm):
        # Computed by angle distance point and satellite ground point
        # Just 10% faster if done by checking normal euclidean distance
        for satellite in sm.satellites:
            norm_sat = norm(satellite.posvel_ecf[0:3])
            satellite.det_lla()
            sat_altitude = norm_sat - misc_fn.earth_radius_lat(satellite.lla[0])
            radius = misc_fn.det_swath_radius(sat_altitude, satellite.obs_incl_angle_stop)
            earth_angle_swath = radians(misc_fn.earth_angle_beta_deg(radius))
            self.user_metric[:,sm.cnt_epoch] = \
            misc_fn.check_users_from_nadir(self.user_metric, self.user_pos, np.array(satellite.posvel_ecf[0:3]),
                                           norm_sat, earth_angle_swath, sm.cnt_epoch)

    def after_loop(self, sm):

        self.plot_swath_coverage(sm, self.user_metric, self.ortho_view_latitude)

        if self.revisit:
            self.compute_plot_revisit(sm, self.user_metric, self.statistic, self.ortho_view_latitude)

        np.save('../output/user_cov_swath', self.user_metric)


class AnalysisObsSwathPushBroom(AnalysisBase, AnalysisObs):

    def __init__(self):
        super().__init__()
        self.ortho_view_latitude = None
        self.revisit = None
        self.statistic = None
        self.planes = np.zeros((4,3))
        self.user_pos = None
        self.user_metric = None

    def read_config(self, node):
        if node.find('OrthoViewLatitude') is not None:
            self.ortho_view_latitude = float(node.find('OrthoViewLatitude').text)
        if node.find('Revisit') is not None:
            self.revisit = misc_fn.str2bool(node.find('Revisit').text)
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text.lower()

    def before_loop(self, sm):
        # Get the incidence angles for each of the satelllites
        for satellite in sm.satellites:
            idx_found = 0
            for idx, constellation in enumerate(sm.constellations):
                if satellite.constellation_id == constellation.constellation_id:
                    idx_found = idx
            const = sm.constellations[idx_found]
            sat_altitude = satellite.kepler.semi_major_axis - R_EARTH  # Note this is an approximation
            if const.obs_swath_start is not None:  # if swath defined by swath length rather than incidence
                satellite.obs_incl_angle_start = misc_fn.incl_from_swath(
                    const.obs_swath_start, R_EARTH, sat_altitude)
                satellite.obs_incl_angle_stop = misc_fn.incl_from_swath(
                    const.obs_swath_stop, R_EARTH, sat_altitude)
            else:
                satellite.obs_incl_angle_start = const.obs_incl_angle_start
                satellite.obs_incl_angle_stop = const.obs_incl_angle_stop
            alfa_critical = asin(R_EARTH / (R_EARTH + sat_altitude))  # If incidence angle shooting off Earth -> error
            if satellite.obs_incl_angle_start > alfa_critical:
                ls.logger.error(f'Inclination angle start: {degrees(satellite.obs_incl_angle_start)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()
            if satellite.obs_incl_angle_stop > alfa_critical:
                ls.logger.error(f'Inclination angle stop: {degrees(satellite.obs_incl_angle_stop)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()

        self.user_pos_ecf = np.zeros((len(sm.users),3))  # User position in ECF
        self.user_metric = np.zeros((len(sm.users), sm.num_epoch),dtype=np.int8)
        for idx_user, user in enumerate(sm.users):
            self.user_pos_ecf[idx_user,:] = np.array(user.posvel_ecf[0:3])

    def in_loop(self, sm):

        for satellite in sm.satellites:
            sat_pos = np.array(satellite.posvel_ecf[0:3])  # Need np array for easy manipulation
            point_vec1 = misc_fn.rot_vec_vec(-sat_pos, np.array(satellite.posvel_ecf[3:6]),
                                             -satellite.obs_incl_angle_start)  # minus for right looking, plus for left
            point_vec2 = misc_fn.rot_vec_vec(-sat_pos, np.array(satellite.posvel_ecf[3:6]),
                                             -satellite.obs_incl_angle_stop)  # minus for right looking, plus for left
            satellite.det_lla()
            r_earth = misc_fn.earth_radius_lat(satellite.lla[0])  # Determine local radius Earth at latitude
            intersect, p1b, satellite.p1 = misc_fn.line_sphere_intersect(sat_pos, sat_pos + point_vec1, r_earth, np.zeros(3))
            intersect, p2b, satellite.p2 = misc_fn.line_sphere_intersect(sat_pos, sat_pos + point_vec2, r_earth, np.zeros(3))
            # 4 Planes of pyramid need to be carefully chosen with normal outwards of pyramid
            self.planes[0,:] = misc_fn.plane_normal(satellite.p1, satellite.p2)
            self.planes[1,:] = misc_fn.plane_normal(satellite.p4, satellite.p3)
            self.planes[2,:] = misc_fn.plane_normal(satellite.p2, satellite.p4)
            self.planes[3,:] = misc_fn.plane_normal(satellite.p3, satellite.p1)
            satellite.p3 = satellite.p1.copy()  # Copy for next run, without .copy() python just refers to same list
            satellite.p4 = satellite.p2.copy()
            if sm.cnt_epoch > 0:  # Now valid point 3 and 4
                self.user_metric[:,sm.cnt_epoch] = misc_fn.check_users_in_plane(self.user_metric, self.user_pos_ecf,
                                                                                self.planes, sm.cnt_epoch)

    def after_loop(self, sm):

        self.plot_swath_coverage(sm, self.user_metric, self.ortho_view_latitude)

        if self.revisit:
            self.compute_plot_revisit(sm, self.user_metric, self.statistic, self.ortho_view_latitude)

        np.save('../output/user_cov_swath', self.user_metric)
