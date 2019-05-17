import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap

import misc_fn
from constants import pi


class AnalysisBase:

    def __init__(self):

        self.times_mjd = []  # Time list of simulation
        self.times_fDOY = []  # Time list of simulation

        self.Type = ''

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

    def before_loop(self,sm):
        for satellite in sm.satellites:
            satellite.Metric = np.zeros((sm.NumEpoch, 3))

    def in_loop(self,sm):
        for satellite in sm.satellites:
            satellite.DetermineLLA()
            satellite.Metric[sm.cnt_epoch, 0] = satellite.LLA[0] / pi * 180
            satellite.Metric[sm.cnt_epoch, 1] = satellite.LLA[1] / pi * 180
            satellite.Metric[sm.cnt_epoch, 2] = satellite.NumStationInView

    def after_loop(self,sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        for satellite in sm.satellites:  # TODO plot ground stations
            plt.scatter(satellite.Metric[:, 1], satellite.Metric[:, 0],
                        c=satellite.Metric[:, 2])
        plt.colorbar(shrink=0.6)
        m = Basemap(projection='cyl', lon_0=0)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.savefig('output/'+self.Type+'.png')
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
            satellite.Metric = np.zeros((sm.NumEpoch, 2))

    def in_loop(self, sm):
        if self.satellite_id > 0:  # Only for one satellite
            for satellite in sm.satellites:
                if satellite.ConstellationID == self.constellation_id and \
                        satellite.SatelliteID == self.satellite_id:
                    satellite.DetermineLLA()
                    satellite.Metric[sm.cnt_epoch, 0] = satellite.LLA[0] / pi * 180
                    satellite.Metric[sm.cnt_epoch, 1] = satellite.LLA[1] / pi * 180
        else:  # Plot the GT for all satellites in the chosen constellation
            for satellite in sm.satellites:
                if satellite.ConstellationID == self.constellation_id:
                    satellite.DetermineLLA()
                    satellite.Metric[sm.cnt_epoch, 0] = satellite.LLA[0] / pi * 180
                    satellite.Metric[sm.cnt_epoch, 1] = satellite.LLA[1] / pi * 180

    def after_loop(self, sm):

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        m = Basemap(projection='cyl', lon_0=0)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        if self.satellite_id > 0:  # Only for one satellite
            for satellite in sm.satellites:
                if satellite.ConstellationID == self.constellation_id and \
                        satellite.SatelliteID == self.satellite_id:
                    y, x = satellite.Metric[:, 0], satellite.Metric[:, 1]
                    plt.plot(x, y, 'r.')
        else:
            for satellite in sm.satellites:
                y, x = satellite.Metric[:, 0], satellite.Metric[:, 1]
                plt.plot(x, y, '+', label=str(satellite.SatelliteID))
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('output/'+self.Type+'.png')
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

    def before_loop(self,sm):
        for user in sm.users:
            user.Metric = np.full((sm.NumEpoch, len(sm.satellites)), False, dtype=bool)

    def in_loop(self,sm):
        for user in sm.users:
            for j in range(user.NumSatInView):
                if sm.satellites[user.IdxSatInView[j]].ConstellationID == self.constellation_id:
                    user.Metric[sm.cnt_epoch, user.IdxSatInView[j]] = True

    def after_loop(self,sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
        lats, lons = [], []
        time_step = int((self.times_mjd[1] - self.times_mjd[0]) * 86400)
        metric = np.zeros(len(sm.users))
        for idx_usr, user in sm.users:
            valid_value_list = []  # Define and clear
            for idx_sat, satellite in sm.satellites:
                for idx_tim in range(1, len(self.times_fDOY)):  # Loop over time (ignoring first)
                    if user.Metric[idx_tim - 1][idx_sat] and not user.Metric[idx_tim][idx_sat]:  # End pass detected
                        length_of_pass = 1  # Compute the length of the pass
                        found_beginning_pass = False
                        while not found_beginning_pass:
                            if idx_tim - length_of_pass >= 0:
                                if user.Metric[idx_tim - length_of_pass][idx_sat]:
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
            lats.append(sm.users[idx_usr].LLA[0] / pi * 180)
            lons.append(sm.users[idx_usr].LLA[1] / pi * 180)

        x_new = np.reshape(np.array(lons), (sm.users[0].NumLat, sm.users[0].NumLon))
        y_new = np.reshape(np.array(lats), (sm.users[0].NumLat, sm.users[0].NumLon))
        z_new = np.reshape(np.array(metric), (sm.users[0].NumLat, sm.users[0].NumLon))
        m = Basemap(projection='cyl', lon_0=0)
        im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        cb = m.colorbar(im1, "right", size="2%", pad="2%")
        cb.set_label(self.statistic + ' Pass Time Interval [s]', fontsize=10)
        plt.savefig('output/'+self.Type+'.png')
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

    def before_loop(self,sm):
        # Find the index of the satellite that is needed
        for i, satellite in enumerate(sm.satellites):
            if satellite.ConstellationID == self.constellation_id and \
                    satellite.SatelliteID == self.satellite_id:
                self.idx_found_satellite = i
                break

    def in_loop(self,sm):
        pass

    def after_loop(self,sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        sm.satellites[self.idx_found_satellite].DetermineLLA()
        contour = misc_fn.SatGrndVis(sm.satellites[self.idx_found_satellite].LLA, self.ElevationMask)
        m = Basemap(projection='cyl', lon_0=0)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.plot(contour[:, 1] / pi * 180, contour[:, 0] / pi * 180, 'r.')
        plt.savefig('output/'+self.Type+'.png')
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
            user.Metric = np.zeros(sm.NumEpoch)

    def in_loop(self, sm):
        for idx_user, user in enumerate(sm.users):
            best_satellite_value = -1
            for idx_sat in range(user.NumSatInView):
                if sm.satellites[user.IdxSatInView[idx_sat]].ConstellationID == self.ConstellationID:
                    elevation = sm.usr2sp[idx_user * len(sm.satellites) + user.IdxSatInView[idx_sat]]. \
                                    Elevation / pi * 180
                    if elevation > best_satellite_value:
                        best_satellite_value = elevation
            user.Metric[sm.cnt_epoch] = best_satellite_value

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
        metric, lats, lons = [], [], []
        for user in sm.users:
            if self.statistic == 'Min':
                metric.append(np.min(user.Metric))
            if self.statistic == 'Mean':
                metric.append(np.mean(user.Metric))
            if self.statistic == 'Max':
                metric.append(np.max(user.Metric))
            if self.statistic == 'Std':
                metric.append(np.std(user.Metric))
            if self.statistic == 'Median':
                metric.append(np.median(user.Metric))
            lats.append(user.LLA[0] / pi * 180)
            lons.append(user.LLA[1] / pi * 180)
        x_new = np.reshape(np.array(lons), (sm.users[0].NumLat, sm.users[0].NumLon))
        y_new = np.reshape(np.array(lats), (sm.users[0].NumLat, sm.users[0].NumLon))
        z_new = np.reshape(np.array(metric), (sm.users[0].NumLat, sm.users[0].NumLon))
        m = Basemap(projection='cyl', lon_0=0)
        im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        cb = m.colorbar(im1, "right", size="2%", pad="2%")
        cb.set_label(self.statistic + ' of Max Elevation satellites in view', fontsize=10)
        plt.savefig('output/'+self.Type+'.png')
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

    def before_loop(self,sm):
        if os.path.exists('output/orbits.txt'):
            os.remove('output/orbits.txt')

    def in_loop(self,sm):
        for idx_sat, satellite in enumerate(sm.satellites):
            if satellite.ConstellationID == self.constellation_id:
                with open('output/orbits.txt', 'a') as f:
                    f.write("%13.6f,%d,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f\n" \
                            % (sm.time_mjd, satellite.SatelliteID,
                               satellite.PosVelECI[0], satellite.PosVelECI[1], satellite.PosVelECI[2],
                               satellite.PosVelECI[3], satellite.PosVelECI[4], satellite.PosVelECI[5]))

    def after_loop(self,sm):
        data = pd.read_csv('output/orbits.txt', sep=',', header=None,
                           names=['RunTime', 'ID', 'x', 'y', 'z', 'x_vel', 'y_vel', 'z_vel'])
        data2 = data[data.ID == 1]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.grid()
        ax1.set_ylabel('Position ECI [m]')
        ax1.plot(self.times_fDOY, data2.x, 'r+-', label='x_pos')
        ax1.plot(self.times_fDOY, data2.y, 'g+-', label='y_pos')
        ax1.plot(self.times_fDOY, data2.z, 'b+-', label='z_pos')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Velocity ECI [m]')  # we already handled the x-label with ax1
        ax2.plot(self.times_fDOY, data2.x_vel, 'm+-', label='x_vel')
        ax2.plot(self.times_fDOY, data2.y_vel, 'y+-', label='y_vel')
        ax2.plot(self.times_fDOY, data2.z_vel, 'k+-', label='z_vel')
        ax1.legend(loc=2); ax2.legend(loc=0)
        plt.xlabel('DOY[-]'); fig.tight_layout()
        plt.savefig('output/'+self.Type+'.png')
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

    def before_loop(self,sm):
        # Find the index of the satellite that is needed
        for i, satellite in enumerate(sm.satellites):
            if satellite.ConstellationID == self.constellation_id and \
                    satellite.SatelliteID == self.satellite_id:
                self.idx_found_satellite = i
                break
        for user in sm.users:
            user.Metric = np.zeros((sm.NumEpoch, 2))

    def in_loop(self,sm):
        num_sat = len(sm.satellites)
        for idx_user, user in enumerate(sm.users):
            if sm.usr2sp[idx_user * num_sat + self.idx_found_satellite].Elevation > 0:
                user.Metric[sm.cnt_epoch, 0] = \
                    sm.usr2sp[idx_user * num_sat + self.idx_found_satellite].Azimuth / pi * 180
                user.Metric[sm.cnt_epoch, 1] = \
                    sm.usr2sp[idx_user * num_sat + self.idx_found_satellite].Elevation / pi * 180

    def after_loop(self,sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        for user in sm.users:
            plt.plot(self.times_fDOY, user.Metric[:, 0], 'r+', label='Azimuth')
            plt.plot(self.times_fDOY, user.Metric[:, 1], 'b+', label='Elevation')
        plt.xlabel('DOY[-]');
        plt.ylabel('Azimuth / Elevation [deg]')
        plt.legend(); plt.grid()
        plt.savefig('output/'+self.Type+'.png')
        plt.show()


class AnalysisCovSatelliteVisible(AnalysisBase):

    def __init__(self):
        super().__init__()

    def read_config(self, file_name):
        pass

    def before_loop(self,sm):
        # Find the index of the user that is needed
        for user in sm.users:
            user.Metric = np.zeros(sm.NumEpoch)

    def in_loop(self,sm):
        for user in sm.users:
            user.Metric[sm.cnt_epoch] = user.NumSatInView

    def after_loop(self,sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        for user in sm.users:  # TODO check multiple users
            plt.plot(self.times_fDOY, user.Metric, 'r-')
        plt.xlabel('DOY[-]');
        plt.ylabel('Number of satellites in view');
        plt.grid()
        plt.savefig('output/'+self.Type+'.png')
        plt.show()


class AnalysisCovSatelliteVisibleGrid(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.statistic = ''  # Mandatory

    def read_config(self, node):
        if node.find('Statistic') is not None:
            self.statistic = int(node.find('Statistic').text)

    def before_loop(self,sm):
        for user in sm.users:
            user.Metric = np.zeros(sm.NumEpoch)

    def in_loop(self,sm):
        for user in sm.users:
            user.Metric[sm.cnt_epoch] = user.NumSatInView

    def after_loop(self,sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
        metric, lats, lons = [], [], []
        for user in sm.users:
            if self.statistic == 'Min':
                metric.append(np.min(user.Metric))
            if self.statistic == 'Mean':
                metric.append(np.mean(user.Metric))
            if self.statistic == 'Max':
                metric.append(np.max(user.Metric))
            if self.statistic == 'Std':
                metric.append(np.std(user.Metric))
            if self.statistic == 'Median':
                metric.append(np.median(user.Metric))
            lats.append(user.LLA[0] / pi * 180)
            lons.append(user.LLA[1] / pi * 180)
        x_new = np.reshape(np.array(lons), (sm.users[0].NumLat, sm.users[0].NumLon))
        y_new = np.reshape(np.array(lats), (sm.users[0].NumLat, sm.users[0].NumLon))
        z_new = np.reshape(np.array(metric), (sm.users[0].NumLat, sm.users[0].NumLon))
        m = Basemap(projection='cyl', lon_0=0)
        im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        cb = m.colorbar(im1, "right", size="2%", pad="2%")
        cb.set_label(self.Statistic + ' Number of satellites in view', fontsize=10)
        plt.savefig('output/'+self.Type+'.png')
        plt.show()


class AnalysisCovSatelliteVisibleId(AnalysisBase):

    def __init__(self):
        super().__init__()

    def read_config(self,file_name):
        pass

    def before_loop(self,sm):
        # Find the index of the user that is needed
        for i in range(len(sm.users)):
            sm.users[i].Metric = np.ones((sm.NumEpoch, len(sm.satellites))) * 999999

    def in_loop(self,sm):
        for idx_sat in range(sm.users[0].NumSatInView):
            if sm.users[0].IdxSatInView[idx_sat] < 999999:
                sm.users[0].Metric[sm.cnt_epoch, idx_sat] = sm.satellites[sm.users[0].IdxSatInView[idx_sat]].SatelliteID

    def after_loop(self,sm):
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        plt.plot(self.times_fDOY, sm.users[0].Metric, 'r+')
        plt.ylim((.5, len(sm.satellites) + 1))
        plt.xlabel('DOY[-]'); plt.ylabel('IDs of satellites in view [-]');
        plt.grid()
        plt.savefig('output/'+self.Type+'.png')
        plt.show()
