import os
import numpy as np
import pandas as pd
from math import degrees, radians

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap

import misc_fn
from constants import PI


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


class AnalysisCovDepthOfCoverage(AnalysisBase):

    def __init__(self):
        super().__init__()

    def read_config(self, node):
        pass

    def before_loop(self, sm):
        for satellite in sm.satellites:
            satellite.metric = np.zeros((sm.num_epoch, 3))

    def in_loop(self, sm):
        for satellite in sm.satellites:
            satellite.det_lla()
            satellite.metric[sm.cnt_epoch, 0] = degrees(satellite.lla[0])
            satellite.metric[sm.cnt_epoch, 1] = degrees(satellite.lla[1])
            satellite.metric[sm.cnt_epoch, 2] = satellite.num_stat_in_view

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        for satellite in sm.satellites:
            plt.scatter(satellite.metric[:, 1], satellite.metric[:, 0],
                        c=satellite.metric[:, 2])
        plt.colorbar(shrink=0.6)
        m = Basemap(projection='cyl', lon_0=0)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        for station in sm.stations:
            plt.plot(degrees(station.lla[1]), degrees(station.lla[0]), 'r^')
        m.drawcoastlines()
        plt.text(50, 80, 'Red triangles: station locations')
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovGroundTrack(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.constellation_id = 0  # Mandatory
        self.satellite_id = 0  # Optional

    def read_config(self, node):
        if node.find('ConstellationID') is not None:
            self.constellation_id = int(node.find('ConstellationID').text)
        if node.find('SatelliteID') is not None:
            self.satellite_id = int(node.find('SatelliteID').text)

    def before_loop(self, sm):
        for satellite in sm.satellites:
            satellite.metric = np.zeros((sm.num_epoch, 2))

    def in_loop(self, sm):
        if self.satellite_id > 0:  # Only for one satellite
            for satellite in sm.satellites:
                if satellite.constellation_id == self.constellation_id and \
                        satellite.sat_id == self.satellite_id:
                    satellite.det_lla()
                    satellite.metric[sm.cnt_epoch, 0] = degrees(satellite.lla[0])
                    satellite.metric[sm.cnt_epoch, 1] = degrees(satellite.lla[1])
        else:  # Plot the GT for all satellites in the chosen constellation
            for satellite in sm.satellites:
                if satellite.constellation_id == self.constellation_id:
                    satellite.det_lla()
                    satellite.metric[sm.cnt_epoch, 0] = degrees(satellite.lla[0])
                    satellite.metric[sm.cnt_epoch, 1] = degrees(satellite.lla[1] )

    def after_loop(self, sm):

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        m = Basemap(projection='cyl', lon_0=0)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        if self.satellite_id > 0:  # Only for one satellite
            for satellite in sm.satellites:
                if satellite.constellation_id == self.constellation_id and \
                        satellite.sat_id == self.satellite_id:
                    y, x = satellite.metric[:, 0], satellite.metric[:, 1]
                    plt.plot(x, y, 'r.')
        else:
            for satellite in sm.satellites:
                y, x = satellite.metric[:, 0], satellite.metric[:, 1]
                plt.plot(x, y, '+', label=str(satellite.sat_id))
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovPassTime(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.constellation_id = 0  # Mandatory
        self.statistic = ''  # Mandatory

    def read_config(self, node):
        if node.find('ConstellationID') is not None:
            self.constellation_id = int(node.find('ConstellationID').text)
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text

    def before_loop(self, sm):
        for user in sm.users:
            user.metric = np.full((sm.num_epoch, len(sm.satellites)), False, dtype=bool)

    def in_loop(self, sm):
        for user in sm.users:
            for j in range(user.num_sat_in_view):
                if sm.satellites[user.idx_sat_in_view[j]].constellation_id == self.constellation_id:
                    user.metric[sm.cnt_epoch, user.idx_sat_in_view[j]] = True

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
        lats, lons = [], []
        time_step = int((self.times_mjd[1] - self.times_mjd[0]) * 86400)
        metric = np.zeros(len(sm.users))
        for idx_usr, user in sm.users:
            valid_value_list = []  # Define and clear
            for idx_sat, satellite in sm.satellites:
                for idx_tim in range(1, len(self.times_f_doy)):  # Loop over time (ignoring first)
                    if user.metric[idx_tim - 1][idx_sat] and not user.metric[idx_tim][idx_sat]:  # End pass detected
                        length_of_pass = 1  # Compute the length of the pass
                        found_beginning_pass = False
                        while not found_beginning_pass:
                            if idx_tim - length_of_pass >= 0:
                                if user.metric[idx_tim - length_of_pass][idx_sat]:
                                    length_of_pass += 1
                                else:
                                    found_beginning_pass = True
                            else:
                                found_beginning_pass = True
                        valid_value_list.append(length_of_pass * time_step)  # Add pass length to the list for this user

            if len(valid_value_list) == 0:
                metric[idx_usr] = -1.0
            else:
                if self.statistic == "Min":
                    metric[idx_usr] = np.min(valid_value_list)
                if self.statistic == "Mean":
                    metric[idx_usr] = np.mean(valid_value_list)
                if self.statistic == "Max":
                    metric[idx_usr] = np.max(valid_value_list)
                if self.statistic == "Std":
                    metric[idx_usr] = np.std(valid_value_list)
                if self.statistic == "Median":
                    metric[idx_usr] = np.median(valid_value_list)
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
        cb.set_label(self.statistic + ' Pass Time Interval [s]', fontsize=10)
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovSatelliteContour(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.constellation_id = 0  # Mandatory
        self.satellite_id = ''  # Mandatory
        self.idx_found_satellite = 0

    def read_config(self, node):
        if node.find('ConstellationID') is not None:
            self.constellation_id = int(node.find('ConstellationID').text)
        if node.find('SatelliteID') is not None:
            self.satellite_id = int(node.find('SatelliteID').text)

    def before_loop(self, sm):
        # Find the index of the satellite that is needed
        for i, satellite in enumerate(sm.satellites):
            if satellite.constellation_id == self.constellation_id and \
                    satellite.sat_id == self.satellite_id:
                self.idx_found_satellite = i
                break

    def in_loop(self, sm):
        pass

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        sm.satellites[self.idx_found_satellite].det_lla()
        contour = misc_fn.sat_contour(sm.satellites[self.idx_found_satellite].lla, self.elevation_mask)
        m = Basemap(projection='cyl', lon_0=0)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.plot(degrees(contour[:, 1]), degrees(contour[:, 0]), 'r.')
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovSatelliteHighest(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.statistic = ''  # Mandatory

    def read_config(self, node):
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text

    def before_loop(self, sm):
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)

    def in_loop(self, sm):
        for idx_user, user in enumerate(sm.users):
            best_satellite_value = -1
            for idx_sat in range(user.num_sat_in_view):
                if sm.satellites[user.idx_sat_in_view[idx_sat]].constellation_id == self.constellation_id:
                    elevation = degrees(sm.usr2sp[idx_user][user.idx_sat_in_view[idx_sat]].elevation)
                    if elevation > best_satellite_value:
                        best_satellite_value = elevation
            user.metric[sm.cnt_epoch] = best_satellite_value

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
        metric, lats, lons = [], [], []
        for user in sm.users:
            if self.statistic == 'Min':
                metric.append(np.min(user.metric))
            if self.statistic == 'Mean':
                metric.append(np.mean(user.metric))
            if self.statistic == 'Max':
                metric.append(np.max(user.metric))
            if self.statistic == 'Std':
                metric.append(np.std(user.metric))
            if self.statistic == 'Median':
                metric.append(np.median(user.metric))
            lats.append(degrees(user.lla[0]))
            lons.append(degrees(user.lla[1]))
        x_new = np.reshape(np.array(lons), (sm.users[0].num_lat, sm.users[0].num_lon))
        y_new = np.reshape(np.array(lats), (sm.users[0].num_lat, sm.users[0].num_lon))
        z_new = np.reshape(np.array(metric), (sm.users[0].num_lat, sm.users[0].num_lon))
        m = Basemap(projection='cyl', lon_0=0)
        im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        cb = m.colorbar(im1, "right", size="2%", pad="2%")
        cb.set_label(self.statistic + ' of Max Elevation satellites in view', fontsize=10)
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovSatellitePvt(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.constellation_id = 0  # Mandatory
        self.satellite_id = ''  # Mandatory

    def read_config(self, node):
        if node.find('ConstellationID') is not None:
            self.constellation_id = int(node.find('ConstellationID').text)
        if node.find('SatelliteID') is not None:
            self.satellite_id = int(node.find('SatelliteID').text)

    def before_loop(self, sm):
        if os.path.exists('output/orbits.txt'):
            os.remove('output/orbits.txt')

    def in_loop(self, sm):
        for idx_sat, satellite in enumerate(sm.satellites):
            if satellite.constellation_id == self.constellation_id:
                with open('output/orbits.txt', 'a') as f:
                    f.write("%13.6f,%d,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f\n" \
                            % (sm.time_mjd, satellite.sat_id,
                               satellite.pvt_eci[0], satellite.pvt_eci[1], satellite.pvt_eci[2],
                               satellite.pvt_eci[3], satellite.pvt_eci[4], satellite.pvt_eci[5]))

    def after_loop(self, sm):
        data = pd.read_csv('output/orbits.txt', sep=',', header=None,
                           names=['RunTime', 'ID', 'x', 'y', 'z', 'x_vel', 'y_vel', 'z_vel'])
        data2 = data[data.ID == 1]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.grid()
        ax1.set_ylabel('Position ECI [m]')
        ax1.plot(self.times_f_doy, data2.x, 'r+-', label='x_pos')
        ax1.plot(self.times_f_doy, data2.y, 'g+-', label='y_pos')
        ax1.plot(self.times_f_doy, data2.z, 'b+-', label='z_pos')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Velocity ECI [m]')  # we already handled the x-label with ax1
        ax2.plot(self.times_f_doy, data2.x_vel, 'm+-', label='x_vel')
        ax2.plot(self.times_f_doy, data2.y_vel, 'y+-', label='y_vel')
        ax2.plot(self.times_f_doy, data2.z_vel, 'k+-', label='z_vel')
        ax1.legend(loc=2); ax2.legend(loc=0)
        plt.xlabel('DOY[-]'); fig.tight_layout()
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovSatelliteSkyAngles(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.constellation_id = 0  # Mandatory
        self.satellite_id = ''  # Mandatory
        self.idx_found_satellite = 0

    def read_config(self, node):
        if node.find('ConstellationID') is not None:
            self.constellation_id = int(node.find('ConstellationID').text)
        if node.find('SatelliteID') is not None:
            self.satellite_id = int(node.find('SatelliteID').text)

    def before_loop(self, sm):
        # Find the index of the satellite that is needed
        for i, satellite in enumerate(sm.satellites):
            if satellite.constellation_id == self.constellation_id and \
                    satellite.sat_id == self.satellite_id:
                self.idx_found_satellite = i
                break
        for user in sm.users:
            user.metric = np.zeros((sm.num_epoch, 2))

    def in_loop(self, sm):
        num_sat = len(sm.satellites)
        for idx_user, user in enumerate(sm.users):
            if sm.usr2sp[idx_user][self.idx_found_satellite].elevation > 0:
                user.metric[sm.cnt_epoch, 0] = degrees(sm.usr2sp[idx_user][self.idx_found_satellite].azimuth)
                user.metric[sm.cnt_epoch, 1] = degrees(sm.usr2sp[idx_user * num_sat + self.idx_found_satellite].elevation)

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        for user in sm.users:
            plt.plot(self.times_f_doy, user.metric[:, 0], 'r+', label='Azimuth')
            plt.plot(self.times_f_doy, user.metric[:, 1], 'b+', label='Elevation')
        plt.xlabel('DOY[-]');
        plt.ylabel('Azimuth / Elevation [deg]')
        plt.legend(); plt.grid()
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovSatelliteVisible(AnalysisBase):

    def __init__(self):
        super().__init__()

    def read_config(self, file_name):
        pass

    def before_loop(self, sm):
        # Find the index of the user that is needed
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)

    def in_loop(self, sm):
        for user in sm.users:
            user.metric[sm.cnt_epoch] = user.num_sat_in_view

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        for user in sm.users:  # TODO check multiple users
            plt.plot(self.times_f_doy, user.metric, 'r-')
        plt.xlabel('DOY[-]');
        plt.ylabel('Number of satellites in view');
        plt.grid()
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovSatelliteVisibleGrid(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.statistic = ''  # Mandatory

    def read_config(self, node):
        if node.find('Statistic') is not None:
            self.statistic = int(node.find('Statistic').text)

    def before_loop(self, sm):
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)

    def in_loop(self, sm):
        for user in sm.users:
            user.metric[sm.cnt_epoch] = user.num_sat_in_view

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
        metric, lats, lons = [], [], []
        for user in sm.users:
            if self.statistic == 'Min':
                metric.append(np.min(user.metric))
            if self.statistic == 'Mean':
                metric.append(np.mean(user.metric))
            if self.statistic == 'Max':
                metric.append(np.max(user.metric))
            if self.statistic == 'Std':
                metric.append(np.std(user.metric))
            if self.statistic == 'Median':
                metric.append(np.median(user.metric))
            lats.append(degrees(user.lla[0]))
            lons.append(degrees(user.lla[1]))
        x_new = np.reshape(np.array(lons), (sm.users[0].num_lat, sm.users[0].num_lon))
        y_new = np.reshape(np.array(lats), (sm.users[0].num_lat, sm.users[0].num_lon))
        z_new = np.reshape(np.array(metric), (sm.users[0].num_lat, sm.users[0].num_lon))
        m = Basemap(projection='cyl', lon_0=0)
        im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        cb = m.colorbar(im1, "right", size="2%", pad="2%")
        cb.set_label(self.Statistic + ' Number of satellites in view', fontsize=10)
        plt.savefig('output/'+self.type+'.png')
        plt.show()


class AnalysisCovSatelliteVisibleId(AnalysisBase):

    def __init__(self):
        super().__init__()

    def read_config(self, file_name):
        pass

    def before_loop(self, sm):
        # Find the index of the user that is needed
        for i in range(len(sm.users)):
            sm.users[i].metric = np.ones((sm.num_epoch, len(sm.satellites))) * 999999

    def in_loop(self, sm):
        for idx_sat in range(sm.users[0].num_sat_in_view):
            if sm.users[0].idx_sat_in_view[idx_sat] < 999999:
                sm.users[0].metric[sm.cnt_epoch, idx_sat] = sm.satellites[sm.users[0].idx_sat_in_view[idx_sat]].sat_id

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        plt.plot(self.times_f_doy, sm.users[0].metric, 'r+')
        plt.ylim((.5, len(sm.satellites) + 1))
        plt.xlabel('DOY[-]'); plt.ylabel('IDs of satellites in view [-]');
        plt.grid()
        plt.savefig('output/'+self.type+'.png')
        plt.show()
