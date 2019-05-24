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


def earth_angle_beta(x):
    beta = degrees(asin(x / R_EARTH))
    return beta


def angle_two_vectors(u,v):
    # Returns angle in [radians]
    u = np.array(u)
    v = np.array(v)
    c = dot(u, v) / norm(u) / norm(v)
    return arccos(clip(c, -1, 1))


class AnalysisObsSwathConical(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.obs_zenith_angle = 0
        self.obs_inclination_angle = 0

    def read_config(self, node):
        if node.find('ObsZenithAngle') is not None:
            self.obs_zenith_angle = radians(float(node.find('ObsZenithAngle').text))
        if node.find('ObsInclinationAngle') is not None:
            self.obs_inclination_angle = radians(float(node.find('ObsInclinationAngle').text))

    def before_loop(self, sm):
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)

    def in_loop(self, sm):
        for satellite in sm.satellites:
            sat_altitude = norm(satellite.posvel_ecf[0:3]) - R_EARTH
            radius = det_swath_radius(sat_altitude, self.obs_inclination_angle)
            earth_angle_swath_deg = earth_angle_beta(radius)
            for user in sm.users:
                angle_user_zenith = angle_two_vectors(user.posvel_ecf[0:3],satellite.posvel_ecf[0:3])
                if angle_user_zenith < radians(earth_angle_swath_deg):
                    user.metric[sm.cnt_epoch] = 1  # Within swath

    def after_loop(self, sm):
        plot_points = []
        for user in sm.users:
            if 1 in user.metric:
                plot_points.append([degrees(user.lla[1]), degrees(user.lla[0])])
        fig = plt.figure(figsize=(10, 4))
        cm = plt.cm.get_cmap('RdYlBu')
        m = Basemap(projection='cyl', lon_0=0)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        x = [row[0] for row in plot_points]
        y = [row[1] for row in plot_points]
        plt.plot(x, y, 'r.',markersize=5, alpha=.2)
        m.drawcoastlines()
        plt.subplots_adjust(left=.01, right=.95, top=0.95, bottom=0.01)
        plt.savefig('output/'+self.type+'.png')
        plt.show()
