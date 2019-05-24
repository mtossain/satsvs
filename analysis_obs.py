import os
import numpy as np
import pandas as pd
from math import degrees, radians
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

# Project modules
import misc_fn
from constants import R_EARTH
from math import tan,sqrt,asin,degrees,radians
from analysis import AnalysisBase


def det_swath_radius(H, inc_angle): # altitude in [m], incidence angle alfa in [radians]
    A = (1 / tan(inc_angle) / tan(inc_angle) + 1)
    B = -2*(H+R_EARTH)
    C = (H+R_EARTH) * (H+R_EARTH) - R_EARTH * R_EARTH / tan(inc_angle) / tan(inc_angle)
    det = sqrt(B*B-4*A*C)
    y1 = (-B + det)/2/A
    x = sqrt(R_EARTH*R_EARTH - y1*y1)
    return x # in [m]


def det_oza(H, inc_angle):  # altitude in [m], incidence angle alfa in [radians]
    x = det_swath_radius(H, inc_angle)
    beta = degrees(asin(x / R_EARTH))
    ia = 90 - degrees(inc_angle) - beta
    oza = 90 - ia
    return oza  # in [deg]


def earth_angle_beta_deg(x):
    # Returns the Earth beta angle in degrees
    beta = degrees(asin(x / R_EARTH))
    return beta


def angle_two_vectors(u,v,norm_u,norm_v):
    # Returns angle in [radians]
    # Pre computed the norm of the second vector
    c = dot(u, v) / norm_u / norm_v
    return arccos(clip(c, -1, 1))


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
            sat_altitude = norm_sat - R_EARTH
            radius = det_swath_radius(sat_altitude, self.obs_inclination_angle)
            earth_angle_swath = radians(earth_angle_beta_deg(radius))
            for user in sm.users:
                angle_user_zenith = angle_two_vectors(user.posvel_ecf[0:3], satellite.posvel_ecf[0:3],
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
