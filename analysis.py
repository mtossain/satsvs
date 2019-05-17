import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap

import misc_fn
import segments
from constants import pi

class Analysis: # TODO Make subclasses for the different analysis


    def __init__(self):

        self.times_mjd = []
        self.times_fDOY = []

        # General analysis parameters
        self.Type = ''

        # Members for specific analysis
        self.ConstellationID = 0
        self.SatelliteID = 0
        self.iFndSatellite = 0
        self.LatitudeRequested = 0.0
        self.LongitudeRequested = 0.0
        self.ElevationMask = 0.0
        self.iFndUser = 0
        self.RequiredNumberSatellites = 0
        self.Statistic = ''  # Depending on analysis it can be Min, Max, Mean, etc.

        # For space to space analysis (Type 20)
        self.TransmitterType = ''  # Type of transmitter (either Satellite or GroundStation)
        self.ReceiverType = ''
        self.TransmitterID = 0
        self.ReceiverID = 0
        self.AmpEff_ef = 0.0
        self.Amp_to_ant_loss_Lf = 0.0
        self.TransmAntGain = []
        self.AntDepointingAngle = 0
        self.RecvAntGain = []
        self.RecvTemp = 0.0
        self.CarrierFreq = 0.0
        self.Modulation = ''
        self.CodeRate = 0.0
        self.MiscLosses = 0.0
        self.AtmObstructionHeight = 0.0  # the height above earth radius to obstruct the link, in m.

        self.PowerInput = 0.0
        self.RecvLineTemp = 0.0  # receiving antenna lines temp
        self.RecvLineLoss = 0.0  # receiving antenna lines losses for noise computation
        self.Bandwidth = 0.0

        # For the rain
        self.RainPExceed = 0.0
        self.RainHeight = 0.0
        self.RainfallRatev = 0.0

        # Type 21
        self.StartPower = 0.0
        self.EndPower = 0.0
        self.Eb_N0 = 0.0
        self.AnalysisDate = 0.0

        # type 22
        self.max_link_quality = 0.0  # to track the maximum non zero in the file
        self.pTransmitterG = segments.Station()
        self.pTransmitterS = segments.Satellite()
        self.pReceiverG = segments.Station()
        self.pReceiverS = segments.Satellite()
        self.worstCaseG2S = segments.Ground2SpaceLink()  # stores range between two terminals (sat or ground) from previous time loop
        self.worstCaseS2S = segments.Space2SpaceLink()
        self.worstCaseS2G = segments.Ground2SpaceLink()

    # General methods for each of the analysis
    def before_loop(self, sm):

        if self.Type == 'cov_ground_track':
            for satellite in sm.satellites:
                satellite.Metric = np.zeros((sm.NumEpoch,2))

        if self.Type == 'cov_satellite_visible':
            # Find the index of the user that is needed
            for user in sm.users:
                user.Metric = np.zeros(sm.NumEpoch)

        if self.Type == 'cov_satellite_visible_grid':
            for user in sm.users:
                user.Metric = np.zeros(sm.NumEpoch)

        if self.Type == 'cov_satellite_visible_id':
            # Find the index of the user that is needed
            for i in range(len(sm.users)):
                sm.users[i].Metric = np.ones((sm.NumEpoch, len(sm.satellites)))*999999

        if self.Type == 'cov_satellite_contour':
            # Find the index of the satellite that is needed
            for i, satellite in enumerate(sm.satellites):
                if satellite.ConstellationID == self.ConstellationID and \
                        satellite.SatelliteID == self.SatelliteID:
                    self.iFndSatellite = i
                    break

        if self.Type == 'cov_satellite_sky_angles':
            # Find the index of the satellite that is needed
            for i, satellite in enumerate(sm.satellites):
                if satellite.ConstellationID == self.ConstellationID and \
                        satellite.SatelliteID == self.SatelliteID:
                    self.iFndSatellite = i
                    break
            for user in sm.users:
                user.Metric = np.zeros((sm.NumEpoch, 2))

        if self.Type == 'cov_depth_of_coverage':
            for satellite in sm.satellites:
                satellite.Metric = np.zeros((sm.NumEpoch,3))

        if self.Type == 'cov_pass_time':
            for user in sm.users:
                user.Metric = np.full((sm.NumEpoch, len(sm.satellites)), False, dtype=bool)

        if self.Type == 'cov_satellite_highest':
            for user in sm.users:
                user.Metric = np.zeros(sm.NumEpoch)

        if self.Type == 'cov_satellite_pvt':
            if os.path.exists('output/orbits.txt'):
                os.remove('output/orbits.txt')

    def in_loop(self, sm):

        if self.Type == 'cov_ground_track':
            if self.SatelliteID > 0:  # Only for one satellite
                for satellite in sm.satellites:
                    if satellite.ConstellationID == self.ConstellationID and \
                            satellite.SatelliteID == self.SatelliteID:
                        satellite.DetermineLLA()
                        satellite.Metric[sm.cnt_epoch, 0] = satellite.LLA[0] / pi * 180
                        satellite.Metric[sm.cnt_epoch, 1] = satellite.LLA[1] / pi * 180
            else:  # Plot the GT for all satellites in the chosen constellation
                for satellite in sm.satellites:
                    if satellite.ConstellationID == self.ConstellationID:
                        satellite.DetermineLLA()
                        satellite.Metric[sm.cnt_epoch, 0] = satellite.LLA[0] / pi * 180
                        satellite.Metric[sm.cnt_epoch, 1] = satellite.LLA[1] / pi * 180

        if self.Type == 'cov_satellite_visible':
            for user in sm.users:
                user.Metric[sm.cnt_epoch] = user.NumSatInView

        if self.Type == 'cov_satellite_visible_grid':
            for user in sm.users:
                user.Metric[sm.cnt_epoch] = user.NumSatInView

        if self.Type == 'cov_satellite_visible_id':  # TODO for a certain constellation
            for idx_sat in range(sm.users[0].NumSatInView):
                if sm.users[0].IdxSatInView[idx_sat] < 999999:
                    sm.users[0].Metric[sm.cnt_epoch, idx_sat] = sm.satellites[sm.users[0].IdxSatInView[idx_sat]].SatelliteID

        if self.Type == 'cov_satellite_sky_angles':  # TODO for spacecraft user test
            num_sat = len(sm.satellites)
            for idx_user,user in enumerate(sm.users):
                if sm.usr2sp[idx_user * num_sat + self.iFndSatellite].Elevation > 0:
                    user.Metric[sm.cnt_epoch, 0] = \
                        sm.usr2sp[idx_user * num_sat + self.iFndSatellite].Azimuth / pi * 180
                    user.Metric[sm.cnt_epoch, 1] = \
                        sm.usr2sp[idx_user * num_sat + self.iFndSatellite].Elevation / pi * 180

        if self.Type == 'cov_depth_of_coverage':
            for satellite in sm.satellites:
                satellite.DetermineLLA()
                satellite.Metric[sm.cnt_epoch, 0] = satellite.LLA[0] / pi * 180
                satellite.Metric[sm.cnt_epoch, 1] = satellite.LLA[1] / pi * 180
                satellite.Metric[sm.cnt_epoch, 2] = satellite.NumStationInView

        if self.Type == 'cov_pass_time':
            for user in sm.users:
                for j in range(user.NumSatInView):
                    if sm.satellites[user.IdxSatInView[j]].ConstellationID == self.ConstellationID:
                        user.Metric[sm.cnt_epoch, user.IdxSatInView[j]] = True

        if self.Type == 'cov_satellite_highest':
            for idx_user, user in enumerate(sm.users):
                best_satellite_value = -1
                for idx_sat in range(user.NumSatInView):
                    if sm.satellites[user.IdxSatInView[idx_sat]].ConstellationID == self.ConstellationID:
                        elevation = sm.usr2sp[idx_user * len(sm.satellites) + user.IdxSatInView[idx_sat]].\
                                        Elevation / pi * 180
                        if elevation > best_satellite_value:
                            best_satellite_value = elevation
                user.Metric[sm.cnt_epoch] = best_satellite_value

        if self.Type == 'cov_satellite_pvt':
            for idx_sat, satellite in enumerate(sm.satellites):
                if satellite.ConstellationID == self.ConstellationID:
                    with open('output/orbits.txt', 'a') as f:
                        f.write("%13.6f,%d,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f\n" % (sm.time_mjd,
                                satellite.SatelliteID, satellite.PosVelECI[0],
                                satellite.PosVelECI[1], satellite.PosVelECI[2],
                                satellite.PosVelECI[3], satellite.PosVelECI[4],
                                satellite.PosVelECI[5]))

    def after_loop(self, sm):

        if self.Type == 'cov_ground_track':
            fig = plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            if self.SatelliteID > 0:  # Only for one satellite
                for satellite in sm.satellites:
                    if satellite.ConstellationID == self.ConstellationID and \
                            satellite.SatelliteID == self.SatelliteID:
                        y, x = satellite.Metric[:, 0], satellite.Metric[:, 1]
                        plt.plot(x, y, 'r.')
            else:
                for satellite in sm.satellites:
                    y, x = satellite.Metric[:, 0], satellite.Metric[:, 1]
                    plt.plot(x, y, '+', label=str(satellite.SatelliteID))
                plt.legend(fontsize=8)
            plt.tight_layout()

        if self.Type == 'cov_satellite_visible':
            fig = plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
            for user in sm.users:  # TODO check multiple users
                plt.plot(self.times_fDOY, user.Metric, 'r-')
            plt.xlabel('DOY[-]'); plt.ylabel('Number of satellites in view'); plt.grid()

        if self.Type == 'cov_satellite_visible_grid':
            fig = plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
            metric, lats, lons = [], [], []
            for user in sm.users:
                if self.Statistic == 'Min':
                    metric.append(np.min(user.Metric))
                if self.Statistic == 'Mean':
                    metric.append(np.mean(user.Metric))
                if self.Statistic == 'Max':
                    metric.append(np.max(user.Metric))
                if self.Statistic == 'Std':
                    metric.append(np.std(user.Metric))
                if self.Statistic == 'Median':
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
            cb.set_label(self.Statistic+' Number of satellites in view', fontsize=10)

        if self.Type == 'cov_satellite_visible_id':
            fig = plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
            plt.plot(self.times_fDOY, sm.users[0].Metric, 'r+')
            plt.ylim((.5, len(sm.satellites)+1))
            plt.xlabel('DOY[-]'); plt.ylabel('IDs of satellites in view [-]'); plt.grid()

        if self.Type == 'cov_satellite_contour':
            fig = plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
            sm.satellites[self.iFndSatellite].DetermineLLA()
            contour = misc_fn.SatGrndVis(sm.satellites[self.iFndSatellite].LLA, self.ElevationMask)
            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            plt.plot(contour[:, 1] / pi * 180, contour[:, 0] / pi * 180, 'r.')

        if self.Type == 'cov_satellite_sky_angles':  # TODO cov_satellite_sky_angles check spacecraft user
            fig = plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
            for user in sm.users:
                plt.plot(self.times_fDOY, user.Metric[:, 0], 'r+', label='Azimuth')
                plt.plot(self.times_fDOY, user.Metric[:, 1], 'b+', label='Elevation')
            plt.xlabel('DOY[-]'); plt.ylabel('Azimuth / Elevation [deg]'); plt.legend(); plt.grid()

        if self.Type == 'cov_depth_of_coverage':  # TODO select satellite/constellation
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

        if self.Type == 'cov_pass_time':
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
                    if self.Statistic == "Min":
                        metric[idx_usr] = np.min(valid_value_list)
                    if self.Statistic == "Mean":
                        metric[idx_usr] = np.mean(valid_value_list)
                    if self.Statistic == "Max":
                        metric[idx_usr] = np.max(valid_value_list)
                    if self.Statistic == "Std":
                        metric[idx_usr] = np.std(valid_value_list)
                    if self.Statistic == "Median":
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
            cb.set_label(self.Statistic+' Pass Time Interval [s]', fontsize=10)

        if self.Type == 'cov_satellite_highest':
            fig = plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
            metric, lats, lons = [], [], []
            for user in sm.users:
                if self.Statistic == 'Min':
                    metric.append(np.min(user.Metric))
                if self.Statistic == 'Mean':
                    metric.append(np.mean(user.Metric))
                if self.Statistic == 'Max':
                    metric.append(np.max(user.Metric))
                if self.Statistic == 'Std':
                    metric.append(np.std(user.Metric))
                if self.Statistic == 'Median':
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
            cb.set_label(self.Statistic+' of Max Elevation satellites in view', fontsize=10)

        if self.Type == 'cov_satellite_pvt':
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

