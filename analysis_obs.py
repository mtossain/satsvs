import os
import numpy as np
import pandas as pd
from math import degrees, radians
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
from numpy.linalg import norm
# Project modules
import misc_fn
from constants import R_EARTH
from math import tan, sqrt, asin, degrees, radians
from analysis import AnalysisBase
import logging_svs as ls


class AnalysisObsSwathConical(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.ortho_view_latitude = 0
        self.obs_inclination_angle = 0

    def read_config(self, node):
        if node.find('ObsInclinationAngle') is not None:
            self.obs_inclination_angle = radians(float(node.find('ObsInclinationAngle').text))
        if node.find('OrthoViewLatitude') is not None:
            self.ortho_view_latitude = float(node.find('OrthoViewLatitude').text)

    def before_loop(self, sm):
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)
            user.norm_ecf = norm(user.posvel_ecf[0:3])

    def in_loop(self, sm):
        for satellite in sm.satellites:
            norm_sat = norm(satellite.posvel_ecf[0:3])
            sat_altitude = norm_sat - R_EARTH  # TODO Compute real radius Earth at latitude
            radius = misc_fn.det_swath_radius(sat_altitude, self.obs_inclination_angle)
            earth_angle_swath = radians(misc_fn.earth_angle_beta_deg(radius))
            for user in sm.users:
                angle_user_zenith = misc_fn.angle_two_vectors(user.posvel_ecf[0:3], satellite.posvel_ecf[0:3],
                                                      user.norm_ecf, norm_sat)
                if angle_user_zenith < earth_angle_swath:
                    user.metric[sm.cnt_epoch] = 1  # Within swath

    def after_loop(self, sm):
        plot_points = []
        for user in sm.users:
            if 1 in user.metric:
                plot_points.append([degrees(user.lla[1]), degrees(user.lla[0])])
        x = [row[0] for row in plot_points]
        y = [row[1] for row in plot_points]
        if self.ortho_view_latitude > 0:
            fig = plt.figure(figsize=(7, 7))
            m = Basemap(projection='ortho', lon_0=0, lat_0=self.ortho_view_latitude, resolution='l')
            x, y = m(x,y)
            m.scatter(x, y, 0.05, marker='o', color='r')
        else:
            fig = plt.figure(figsize=(10, 5))
            m = Basemap(projection='cyl', lon_0=0)
            plt.plot(x, y, 'rs', markersize=5, alpha=.3)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.subplots_adjust(left=.1, right=.9, top=0.92, bottom=0.1)
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisObsSwathPushBroom(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.ortho_view_latitude = 0
        self.p1 = [0,0,0]
        self.p2 = [0,0,0]
        self.p3 = [0,0,0]
        self.p4 = [0,0,0]
        self.obs_inclination_angle1 = 0
        self.obs_inclination_angle2 = 0
        self.obs_swath_start = 0.0  # start from nadir in meters
        self.obs_swath_stop = 0.0  # stop from nadir in meters

    def read_config(self, node):
        if node.find('ObsInclinationAngle1') is not None:
            self.obs_inclination_angle1 = radians(float(node.find('ObsInclinationAngle1').text))
        if node.find('ObsInclinationAngle2') is not None:
            self.obs_inclination_angle2 = radians(float(node.find('ObsInclinationAngle2').text))
        if node.find('ObsSwathStart') is not None:
            self.obs_swath_start = float(node.find('ObsSwathStart').text)
        if node.find('ObsSwathStop') is not None:
            self.obs_swath_stop = float(node.find('ObsSwathStop').text)
        if node.find('OrthoViewLatitude') is not None:
            self.ortho_view_latitude = float(node.find('OrthoViewLatitude').text)

    def before_loop(self, sm):
        sat_altitude = sm.satellites[0].kepler.semi_major_axis - R_EARTH
        if self.obs_swath_start > 0:
            self.obs_inclination_angle1 = misc_fn.solve_inc_angle_from_swath_width(
                self.obs_swath_start, R_EARTH, sat_altitude)
            self.obs_inclination_angle2 = misc_fn.solve_inc_angle_from_swath_width(
                self.obs_swath_stop, R_EARTH, sat_altitude)
        alfa_critical = asin(R_EARTH / (R_EARTH + sat_altitude))
        if self.obs_inclination_angle1 > alfa_critical:
            ls.logger.error(f'Inclination angle 1: {degrees(self.obs_inclination_angle1)} ' +
                            f'larger than critical angle {round(degrees(alfa_critical),1)}')
            exit()
        if self.obs_inclination_angle2 > alfa_critical:
            ls.logger.error(f'Inclination angle 2: {degrees(self.obs_inclination_angle2)} ' +
                            f'larger than critical angle {round(degrees(alfa_critical),1)}')
            exit()
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)
            user.norm_ecf = norm(user.posvel_ecf[0:3])

    def in_loop(self, sm):

        planes = []
        for satellite in sm.satellites:
            sat_pos = np.array(satellite.posvel_ecf[0:3])  # Need np array for easy manipulation
            pointing_vec1 = misc_fn.rotate_vec_about_vec(-sat_pos, np.array(satellite.posvel_ecf[3:6]),
                                                 -self.obs_inclination_angle1)  # minus for right looking, plus for left
            pointing_vec2 = misc_fn.rotate_vec_about_vec(-sat_pos, np.array(satellite.posvel_ecf[3:6]),
                                                 -self.obs_inclination_angle2)  # minus for right looking, plus for left
            intersect, p1b, self.p1 = misc_fn.line_sphere_intersect(sat_pos, sat_pos+pointing_vec1, R_EARTH, [0, 0, 0])
            intersect, p2b, self.p2 = misc_fn.line_sphere_intersect(sat_pos, sat_pos+pointing_vec2, R_EARTH, [0, 0, 0])
            # 4 Planes of pyramid need to be carefully choosen with normal outwards of pyramid
            planes.append(misc_fn.Plane(np.array([0, 0, 0]), np.array(self.p1), np.array(self.p2)))
            planes.append(misc_fn.Plane(np.array([0, 0, 0]), np.array(self.p4), np.array(self.p3)))
            planes.append(misc_fn.Plane(np.array([0, 0, 0]), np.array(self.p2), np.array(self.p4)))
            planes.append(misc_fn.Plane(np.array([0, 0, 0]), np.array(self.p3), np.array(self.p1)))
            self.p3 = self.p1  # Copy for next run
            self.p4 = self.p2  # Copy for next run
            if sm.cnt_epoch > 0:  # Now valid point 3 and 4
                for user in sm.users:
                    if misc_fn.test_point_within_pyramid(np.array(user.posvel_ecf[0:3]), planes):
                        user.metric[sm.cnt_epoch] = 1  # Within swath

    def after_loop(self, sm):
        plot_points = []
        for user in sm.users:
            if 1 in user.metric:
                plot_points.append([degrees(user.lla[1]), degrees(user.lla[0])])
        x = [row[0] for row in plot_points]
        y = [row[1] for row in plot_points]
        if self.ortho_view_latitude > 0:
            fig = plt.figure(figsize=(7, 7))
            m = Basemap(projection='ortho', lon_0=0, lat_0=self.ortho_view_latitude, resolution='l')
            x, y = m(x,y)
            m.scatter(x, y, 0.05, marker='o', color='r')
        else:
            fig = plt.figure(figsize=(10, 5))
            m = Basemap(projection='cyl', lon_0=0)
            plt.plot(x, y, 'rs', markersize=3, alpha=.2)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.subplots_adjust(left=.1, right=.9, top=0.92, bottom=0.1)
        plt.savefig('output/'+self.type+'.png')
        plt.show()

