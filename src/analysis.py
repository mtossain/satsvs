import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
from math import sin, cos, asin, degrees, radians
import numpy as np
from numba import jit
# Modules from project
import logging_svs as ls


class AnalysisBase:

    def __init__(self):

        self.times_mjd = []  # Time list of simulation
        self.times_f_doy = []  # Time list of simulation

        self.type = ''

    def read_config(self, node):  # Node in xml element tree
        pass

    def before_loop(self, sm):
        pass

    def in_loop(self, sm):
        pass

    def after_loop(self, sm):
        pass


class AnalysisObs:   # Common methods needed for some OBS analysis

    def plot_swath_coverage(self, sm, user_metric, polar_view):
        plot_points = np.zeros((len(sm.users), 3))
        for idx_user, user in enumerate(sm.users):
            if idx_user % 1000 == 0:
                ls.logger.info(f'User swath coverage {user.user_id} of {len(sm.users)}')
            if user_metric[idx_user, :].any():  # Any value bigger than 0
                num_swaths = len(np.flatnonzero(np.diff(user_metric[idx_user, :]))) / 2
                plot_points[idx_user, :] = [degrees(user.lla[1]), degrees(user.lla[0]), num_swaths]
        plot_points = plot_points[~np.all(plot_points == 0, axis=1)]  # Clean up empty rows
        if polar_view is not None:
            fig = plt.figure(figsize=(6, 6))
            m = Basemap(projection='npstere', lon_0=0, boundinglat=polar_view, resolution='l')
            x, y = m(plot_points[:,0], plot_points[:,1])
            sc = m.scatter(x, y, s=3, marker='o', cmap=plt.cm.jet, c=plot_points[:,2], alpha=.3)
        else:
            fig = plt.figure(figsize=(10, 5))
            m = Basemap(projection='cyl', lon_0=0)
            x, y = plot_points[:,0], plot_points[:,1]
            sc = m.scatter(x, y, s=3, marker='o', cmap=plt.cm.jet, c=plot_points[:,2], alpha=.3)
        cb = m.colorbar(sc, shrink=0.85)
        cb.set_label('Number of passes [-]', fontsize=10)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.subplots_adjust(left=.1, right=.9, top=0.92, bottom=0.1)
        plt.savefig('../output/'+self.type+'.png')
        plt.show()

    def plot_swath_revisit(self, sm, user_metric, statistic, polar_view):
        plot_points = np.zeros((len(sm.users), 3))
        metric = 0
        for idx_user, user in enumerate(sm.users):
            if idx_user % 1000 == 0:
                ls.logger.info(f'User revisit {user.user_id} of {len(sm.users)}')
            gaps = np.diff(np.where(np.diff(user_metric[idx_user,:]) != 0)).flatten()
            if len(gaps) > 1:
                gaps = np.delete(gaps, np.where(gaps == 1), axis=0) * sm.time_step / 3600.0
                if statistic == "min":
                    metric = (np.min(gaps))
                if statistic == "mean":
                    metric = (np.mean(gaps))
                if statistic == "max":
                    metric = (np.max(gaps))
                if statistic == "std":
                    metric = (np.std(gaps))
                if statistic == "median":
                    metric = (np.median(gaps))
                plot_points[idx_user, :] = [degrees(sm.users[idx_user].lla[1]), degrees(sm.users[idx_user].lla[0]), metric]
        plot_points = plot_points[~np.all(plot_points == 0, axis=1)]  # Clean up empty rows
        if metric>0:  # Only plot if not empty
            if polar_view is not None:
                fig = plt.figure(figsize=(6, 6))
                m = Basemap(projection='npstere', lon_0=0, boundinglat=polar_view, resolution='l')
                x, y = m(plot_points[:,0], plot_points[:,1])
                sc = m.scatter(x, y, s=5, cmap=plt.cm.jet, c=plot_points[:,2],
                               vmin=np.min(plot_points[:,2]), vmax=np.max(plot_points[:,2]))
            else:
                fig = plt.figure(figsize=(10, 5))
                m = Basemap(projection='cyl', lon_0=0)
                x, y = plot_points[:, 0], plot_points[:, 1]
                sc = m.scatter(x, y, cmap=plt.cm.jet, s=3, c=plot_points[:,2],
                               vmin=np.min(plot_points[:,2]), vmax=np.max(plot_points[:,2]))
            cb = m.colorbar(sc, shrink=0.85)
            cb.set_label(statistic.capitalize() + ' Revisit Interval [hours]', fontsize=10)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            plt.subplots_adjust(left=.1, right=.9, top=0.9, bottom=0.1)
            plt.savefig(f'../output/{self.type}_revisit.png')
            plt.show()

