import os
import numpy as np
import pandas as pd
from math import degrees, radians
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
# Import project modules
from src import misc_fn
from src.constants import PI
from src.analysis import AnalysisBase


class AnalysisNavDOP(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.direction = None  # Mandatory
        self.statistic = None  # Mandatory

    def read_config(self, node):
        if node.find('Direction') is not None:
            self.direction = node.find('Direction').text
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text

    def before_loop(self, sm):
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)

    def in_loop(self, sm):
        for user_idx, user in enumerate(sm.users):
            if len(user.idx_sat_in_view) > 3:
                h_mat = np.ones((len(user.idx_sat_in_view),4))
                for i, idx_sat in enumerate(user.idx_sat_in_view):
                    h_mat[i,0:3] = sm.usr2sp[user_idx][idx_sat].usr2sp_ecf/sm.usr2sp[user_idx][idx_sat].distance
                hth_inv = misc_fn.inverse4by4(np.matmul(np.transpose(h_mat),h_mat))  # Fast implementation
                #hth_inv2 = np.linalg.inv(np.matmul(np.transpose(h_mat),h_mat))
                q = misc_fn.ecef2enu(hth_inv,user.lla[0],user.lla[1])
                if self.direction == "Pos":
                    dop = np.sqrt(q[0, 0] + q[1, 1] + q[2, 2])
                elif self.direction == "Hor":
                    dop = np.sqrt(q[0, 0] + q[1, 1])
                elif self.direction == "Ver":
                    dop = np.sqrt(q[2, 2])
                user.metric[sm.cnt_epoch] = dop
            else:
                user.metric[sm.cnt_epoch] = np.nan

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 5))
        lats, lons = [], []
        metric = np.zeros(len(sm.users))
        for idx_usr, user in enumerate(sm.users):
            if self.statistic == "Min":
                metric[idx_usr] = np.nanmin(user.metric)
            if self.statistic == "Mean":
                metric[idx_usr] = np.nanmean(user.metric)
            if self.statistic == "Max":
                metric[idx_usr] = np.nanmax(user.metric)
            if self.statistic == "Std":
                metric[idx_usr] = np.nanstd(user.metric)
            if self.statistic == "Median":
                metric[idx_usr] = np.nanmedian(user.metric)
            lats.append(degrees(sm.users[idx_usr].lla[0]))
            lons.append(degrees(sm.users[idx_usr].lla[1]))

        x_new = np.reshape(np.array(lons), (sm.users[0].num_lat, sm.users[0].num_lon))
        y_new = np.reshape(np.array(lats), (sm.users[0].num_lat, sm.users[0].num_lon))
        z_new = np.reshape(np.array(metric), (sm.users[0].num_lat, sm.users[0].num_lon))
        m = Basemap(projection='cyl', lon_0=0)
        im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        cb = m.colorbar(im1, "right", size="2%", pad="2%")
        cb.set_label(self.statistic + ' ' + self.direction + ' Dilution of Precision [-]', fontsize=10)
        plt.subplots_adjust(left=.1, right=.9, top=0.9, bottom=0.1)
        plt.savefig('../output/'+self.type+'.png')
        plt.show()

class AnalysisNavAccuracy(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.direction = None  # Mandatory
        self.statistic = None  # Mandatory

    def read_config(self, node):
        if node.find('Direction') is not None:
            self.direction = node.find('Direction').text
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text

    def before_loop(self, sm):
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)

    def in_loop(self, sm):
        for user_idx, user in enumerate(sm.users):
            sum_user_error = 0
            if len(user.idx_sat_in_view) > 3:
                h_mat = np.ones((len(user.idx_sat_in_view),4))
                for i, idx_sat in enumerate(user.idx_sat_in_view):
                    h_mat[i,0:3] = sm.usr2sp[user_idx][idx_sat].usr2sp_ecf/sm.usr2sp[user_idx][idx_sat].distance
                    num_uere = len(sm.satellites[idx_sat].uere_list)
                    el_piece_angle = PI/2/num_uere
                    idx_el = int(np.floor(sm.usr2sp[user_idx][idx_sat].elevation/el_piece_angle))
                    uere = np.power(sm.satellites[idx_sat].uere_list[idx_el],2)
                    sum_user_error += uere
                hth_inv = misc_fn.inverse4by4(np.matmul(np.transpose(h_mat),h_mat))  # Fast implementation
                #hth_inv = np.linalg.inv(np.matmul(np.transpose(h_mat),h_mat))
                q = misc_fn.ecef2enu(hth_inv,user.lla[0],user.lla[1])
                error = 2 * np.sqrt(sum_user_error/len(user.idx_sat_in_view))
                if self.direction == "Pos":
                    acc = error * np.sqrt(q[0, 0] + q[1, 1] + q[2, 2])
                elif self.direction == "Hor":
                    acc = error * np.sqrt(q[0, 0] + q[1, 1])
                elif self.direction == "Ver":
                    acc = error * np.sqrt(q[2, 2])
                user.metric[sm.cnt_epoch] = acc
            else:
                user.metric[sm.cnt_epoch] = np.nan

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 5))
        lats, lons = [], []
        metric = np.zeros(len(sm.users))
        for idx_usr, user in enumerate(sm.users):
            if self.statistic == "Min":
                metric[idx_usr] = np.nanmin(user.metric)
            if self.statistic == "Mean":
                metric[idx_usr] = np.nanmean(user.metric)
            if self.statistic == "Max":
                metric[idx_usr] = np.nanmax(user.metric)
            if self.statistic == "Std":
                metric[idx_usr] = np.nanstd(user.metric)
            if self.statistic == "Median":
                metric[idx_usr] = np.nanmedian(user.metric)
            lats.append(degrees(sm.users[idx_usr].lla[0]))
            lons.append(degrees(sm.users[idx_usr].lla[1]))

        x_new = np.reshape(np.array(lons), (sm.users[0].num_lat, sm.users[0].num_lon))
        y_new = np.reshape(np.array(lats), (sm.users[0].num_lat, sm.users[0].num_lon))
        z_new = np.reshape(np.array(metric), (sm.users[0].num_lat, sm.users[0].num_lon))
        m = Basemap(projection='cyl', lon_0=0)
        im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        cb = m.colorbar(im1, "right", size="2%", pad="2%")
        cb.set_label(self.statistic + ' ' + self.direction + ' Navigation Accuracy 95% [m]', fontsize=10)
        plt.subplots_adjust(left=.1, right=.9, top=0.9, bottom=0.1)
        plt.savefig('../output/'+self.type+'.png')
        plt.show()



