import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
from math import sin, cos, asin, degrees, radians
import numpy as np
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

    def plot_swath_coverage(self, sm, user_metric, ortho_view_latitude):
        plot_points = []
        for idx_user,user in enumerate(sm.users):
            if idx_user % 1000 == 0:
                ls.logger.info(f'User swath coverage {user.user_id} of {len(sm.users)}')
            if user_metric[idx_user,:].any():  # Any value bigger than 0
                num_swaths = list(np.diff(user_metric[idx_user,:])).count(1)
                plot_points.append([degrees(user.lla[1]), degrees(user.lla[0]), num_swaths])
        x = [row[0] for row in plot_points]
        y = [row[1] for row in plot_points]
        z = [row[2] for row in plot_points]
        if ortho_view_latitude is not None:
            fig = plt.figure(figsize=(7, 7))
            m = Basemap(projection='npstere', lon_0=0, boundinglat=ortho_view_latitude, resolution='l')
            x, y = m(x, y)
            sc = m.scatter(x, y, s=3, marker='o', cmap=plt.cm.jet, c=z, alpha=.3)
        else:
            fig = plt.figure(figsize=(10, 5))
            m = Basemap(projection='cyl', lon_0=0)
            sc = m.scatter(x, y, s=3, marker='o', cmap=plt.cm.jet, c=z, alpha=.3)
        cb = m.colorbar(sc, shrink=0.85)
        cb.set_label('Number of passes [-]', fontsize=10)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.subplots_adjust(left=.1, right=.9, top=0.92, bottom=0.1)
        plt.savefig('../output/'+self.type+'.png')
        plt.show()

    def compute_plot_revisit(self, sm, user_metric, statistic, ortho_view_latitude):
        lats, lons, metric = [], [], []
        for idx_user, user in enumerate(sm.users):
            if idx_user % 1000 == 0:
                ls.logger.info(f'User revisit {user.user_id} of {len(sm.users)}')
            gaps = np.diff(np.where(np.diff(user_metric[idx_user,:]) != 0)).flatten()
            if len(gaps)>1:
                gaps = np.delete(gaps, np.where(gaps == 1), axis=0) * sm.time_step / 3600.0

                if statistic == "min":
                    metric.append(np.min(gaps))
                if statistic == "mean":
                    metric.append(np.mean(gaps))
                if statistic == "max":
                    metric.append(np.max(gaps))
                if statistic == "std":
                    metric.append(np.std(gaps))
                if statistic == "median":
                    metric.append(np.median(gaps))
                lats.append(degrees(sm.users[idx_user].lla[0]))
                lons.append(degrees(sm.users[idx_user].lla[1]))

        if lats:  # Only plot if not empty
            if ortho_view_latitude is not None:
                fig = plt.figure(figsize=(7, 7))
                m = Basemap(projection='npstere', lon_0=0, boundinglat=ortho_view_latitude, resolution='l')
                lons, lats = m(lons,lats)
                sc = m.scatter(lons, lats, s=5, cmap=plt.cm.jet, c=metric, vmin=0, vmax=np.max(metric))
            else:
                fig = plt.figure(figsize=(10, 5))
                m = Basemap(projection='cyl', lon_0=0)
                sc = m.scatter(lons, lats, cmap=plt.cm.jet, s=3, c=metric, vmin=0, vmax=np.max(metric))
            cb = m.colorbar(sc, shrink=0.85)
            cb.set_label(statistic.capitalize() + ' Revisit Interval [hours]', fontsize=10)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            plt.subplots_adjust(left=.1, right=.9, top=0.9, bottom=0.1)
            plt.savefig(f'../output/{self.type}_revisit.png')
            plt.show()

