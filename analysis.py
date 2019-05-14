
import matplotlib
import os
import numpy as np
matplotlib.use("TkAgg")  # TODO solve bug with basemap???
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
import pandas as pd

import misc_fn
import segments
from constants import pi

class Analysis: # TODO Make subclasses for the different analysis

    TimeListISO = []  # Class variables
    TimeListMJD = []
    TimeListfDOY = []

    def __init__(self):

        # General analysis parameters
        self.Type = ''

        # Members for specific analysis
        self.ConstellationID = 0
        self.ConstellationName = ''
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
        self.pTransmitterG = segments.GroundStation()
        self.pTransmitterS = segments.Satellite()
        self.pReceiverG = segments.GroundStation()
        self.pReceiverS = segments.Satellite()
        self.worstCaseG2S = segments.Ground2SpaceLink()  # stores range between two terminals (sat or ground) from previous time loop
        self.worstCaseS2S = segments.Space2SpaceLink()
        self.worstCaseS2G = segments.Ground2SpaceLink()
        self.AnalysisFigureOfMerit = []  # vector containing the statistic for every user
        self.AnalysisMemory2D = []  # To save space a int is used(16bit)
        self.AnalysisMemory2DBool = []  # To save space a bool is used (1bit)
        self.AnalysisMemory3DBool = []  # To save space a bool is used (1bit)
        self.AnalysisMemory2DShort = []  # To save space a short int is used (8bit)

    # General methods for each of the analysis
    def RunAnalysisBeforeTimeLoop(self, NumEpoch, TimeStep, SatelliteList, UserList, AnalysisList):

        if self.Type == 'cov_ground_track':
            for i in range(len(SatelliteList)):
                SatelliteList[i].Metric = np.zeros((NumEpoch,2))

        if self.Type == 'cov_satellite_visible':
            # Find the index of the user that is needed
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros(NumEpoch)

        if self.Type == 'cov_satellite_visible_grid':
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros(NumEpoch)

        if self.Type == 'cov_satellite_visible_id':
            # Find the index of the user that is needed
            for i in range(len(UserList)):
                UserList[i].Metric = np.ones((NumEpoch, len(SatelliteList)))*999999

        if self.Type == 'cov_satellite_contour':
            # Find the index of the satellite that is needed
            for i in range(len(SatelliteList)):
                if SatelliteList[i].ConstellationID == self.ConstellationID and \
                        SatelliteList[i].SatelliteID == self.SatelliteID:
                    self.iFndSatellite = i
                    break

        if self.Type == 'cov_satellite_sky_angles':
            # Find the index of the satellite that is needed
            for i in range(len(SatelliteList)):
                if SatelliteList[i].ConstellationID == self.ConstellationID and \
                        SatelliteList[i].SatelliteID == self.SatelliteID:
                    self.iFndSatellite = i
                    break
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros((NumEpoch, 2))

        if self.Type == 'cov_depth_of_coverage':
            for i in range(len(SatelliteList)):
                SatelliteList[i].Metric = np.zeros((NumEpoch,3))

        if self.Type == 'cov_pass_time':
            for i in range(len(UserList)):
                UserList[i].Metric = np.full((NumEpoch, len(SatelliteList)), False, dtype=bool)

        if self.Type == 'cov_satellite_highest':
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros(NumEpoch)

        if self.Type == 'cov_satellite_pvt':
            try:
                os.remove('output/orbits.txt')
            except:
                pass

    def RunAnalysisInTimeLoop(self, RunTime, CntEpoch, SatelliteList, UserList, User2SatelliteList):

        if self.Type == 'cov_ground_track':
            if self.SatelliteID > 0:  # Only for one satellite
                for idx in range(len(SatelliteList)):
                    if SatelliteList[idx].ConstellationID == self.ConstellationID and \
                            SatelliteList[idx].SatelliteID == self.SatelliteID:
                        SatelliteList[idx].DetermineLLA()
                        SatelliteList[idx].Metric[CntEpoch, 0] = SatelliteList[idx].LLA[0] / pi * 180
                        SatelliteList[idx].Metric[CntEpoch, 1] = SatelliteList[idx].LLA[1] / pi * 180
            else:  # Plot the GT for all satellites in the chosen constellation
                for idx in range(len(SatelliteList)):
                    if SatelliteList[idx].ConstellationID == self.ConstellationID:
                        SatelliteList[idx].DetermineLLA()
                        SatelliteList[idx].Metric[CntEpoch, 0] = SatelliteList[idx].LLA[0] / pi * 180
                        SatelliteList[idx].Metric[CntEpoch, 1] = SatelliteList[idx].LLA[1] / pi * 180

        if self.Type == 'cov_satellite_visible':
            for idx_user in range(len(UserList)):
                UserList[idx_user].Metric[CntEpoch] = UserList[idx_user].NumSatInView

        if self.Type == 'cov_satellite_visible_grid':
            for idx_user in range(len(UserList)):
                UserList[idx_user].Metric[CntEpoch] = UserList[idx_user].NumSatInView

        if self.Type == 'cov_satellite_visible_id':  # TODO for a certain constellation
            for idx_sat in range(UserList[0].NumSatInView):
                if UserList[0].IdxSatInView[idx_sat] < 999999:
                    UserList[0].Metric[CntEpoch, idx_sat] = SatelliteList[UserList[0].IdxSatInView[idx_sat]].SatelliteID

        if self.Type == 'cov_satellite_sky_angles':  # TODO for spacecraft user test
            num_sat = len(SatelliteList)
            for idx_user in range(len(UserList)):
                if User2SatelliteList[idx_user * num_sat + self.iFndSatellite].Elevation > 0:
                    UserList[idx_user].Metric[CntEpoch, 0] = \
                        User2SatelliteList[idx_user * num_sat + self.iFndSatellite].Azimuth / pi * 180
                    UserList[idx_user].Metric[CntEpoch, 1] = \
                        User2SatelliteList[idx_user * num_sat + self.iFndSatellite].Elevation / pi * 180

        if self.Type == 'cov_depth_of_coverage':
            for idx in range(len(SatelliteList)):
                SatelliteList[idx].DetermineLLA()
                SatelliteList[idx].Metric[CntEpoch, 0] = SatelliteList[idx].LLA[0] / pi * 180
                SatelliteList[idx].Metric[CntEpoch, 1] = SatelliteList[idx].LLA[1] / pi * 180
                SatelliteList[idx].Metric[CntEpoch, 2] = SatelliteList[idx].NumStationInView

        if self.Type == 'cov_pass_time':
            for idx in range(len(UserList)):
                for j in range(UserList[idx].NumSatInView):
                    if SatelliteList[UserList[idx].IdxSatInView[j]].ConstellationID == self.ConstellationID:
                        UserList[idx].Metric[CntEpoch, UserList[idx].IdxSatInView[j]] = True

        if self.Type == 'cov_satellite_highest':
            for idx_user in range(len(UserList)):
                best_satellite_value = -1
                for idx_sat in range(UserList[idx_user].NumSatInView):
                    if SatelliteList[UserList[idx_user].IdxSatInView[idx_sat]].ConstellationID == self.ConstellationID:
                        elevation = User2SatelliteList[idx_user * len(SatelliteList) + UserList[idx_user].IdxSatInView[idx_sat]].\
                                        Elevation / pi * 180
                        if elevation > best_satellite_value:
                            best_satellite_value = elevation
                UserList[idx_user].Metric[CntEpoch] = best_satellite_value

        if self.Type == 'cov_satellite_pvt':
            for idx_sat in range(len(SatelliteList)):
                if SatelliteList[idx_sat].ConstellationID == self.ConstellationID:
                    with open('output/orbits.txt', 'a') as f:
                        f.write("%13.6f,%d,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f\n" % (RunTime,
                                SatelliteList[idx_sat].SatelliteID, SatelliteList[idx_sat].PosVelECI[0],
                                SatelliteList[idx_sat].PosVelECI[1], SatelliteList[idx_sat].PosVelECI[2],
                                SatelliteList[idx_sat].PosVelECI[3], SatelliteList[idx_sat].PosVelECI[4],
                                SatelliteList[idx_sat].PosVelECI[5]))


    def RunAnalysisAfterTimeLoop(self, SatelliteList, UserList):

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)

        if self.Type == 'cov_ground_track':
            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            if self.SatelliteID > 0:  # Only for one satellite
                for i in range(len(SatelliteList)):
                    if SatelliteList[i].ConstellationID == self.ConstellationID and \
                            SatelliteList[i].SatelliteID == self.SatelliteID:
                        y, x = SatelliteList[i].Metric[:, 0], SatelliteList[i].Metric[:, 1]
                        plt.plot(x, y, 'r.')
            else:
                for i in range(len(SatelliteList)):
                    y, x = SatelliteList[i].Metric[:, 0], SatelliteList[i].Metric[:, 1]
                    plt.plot(x, y, '+', label=str(SatelliteList[i].SatelliteID))
                plt.legend(fontsize=8)
            plt.tight_layout()

        if self.Type == 'cov_satellite_visible':
            for i in range(len(UserList)):  # TODO check multiple users
                plt.plot(Analysis.TimeListfDOY, UserList[i].Metric, 'r-')
            plt.xlabel('DOY[-]'); plt.ylabel('Number of satellites in view'); plt.grid()

        if self.Type == 'cov_satellite_visible_grid':
            plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
            metric, lats, lons = [], [], []
            for i in range(len(UserList)):
                if self.Statistic == 'Min':
                    metric.append(np.min(UserList[i].Metric))
                if self.Statistic == 'Mean':
                    metric.append(np.mean(UserList[i].Metric))
                if self.Statistic == 'Max':
                    metric.append(np.max(UserList[i].Metric))
                if self.Statistic == 'Std':
                    metric.append(np.std(UserList[i].Metric))
                if self.Statistic == 'Median':
                    metric.append(np.median(UserList[i].Metric))
                lats.append(UserList[i].LLA[0] / pi * 180)
                lons.append(UserList[i].LLA[1] / pi * 180)
            x_new = np.reshape(np.array(lons), (UserList[0].NumLat,UserList[0].NumLon))
            y_new = np.reshape(np.array(lats), (UserList[0].NumLat,UserList[0].NumLon))
            z_new = np.reshape(np.array(metric), (UserList[0].NumLat,UserList[0].NumLon))
            m = Basemap(projection='cyl', lon_0=0)
            im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            cb = m.colorbar(im1, "right", size="2%", pad="2%")
            cb.set_label(self.Statistic+' Number of satellites in view', fontsize=10)

        if self.Type == 'cov_satellite_visible_id':
            plt.plot(Analysis.TimeListfDOY, UserList[0].Metric, 'r+')
            plt.ylim((.5, len(SatelliteList)+1))
            plt.xlabel('DOY[-]'); plt.ylabel('IDs of satellites in view [-]'); plt.grid()

        if self.Type == 'cov_satellite_contour':
            SatelliteList[self.iFndSatellite].DetermineLLA()
            contour = misc_fn.SatGrndVis(SatelliteList[self.iFndSatellite].LLA, self.ElevationMask)
            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            plt.plot(contour[:, 1] / pi * 180, contour[:, 0] / pi * 180, 'r.')

        if self.Type == 'cov_satellite_sky_angles':  # TODO cov_satellite_sky_angles check spacecraft user
            for idx_user in range(len(UserList)):
                plt.plot(Analysis.TimeListfDOY, UserList[idx_user].Metric[:, 0], 'r+', label='Azimuth')
                plt.plot(Analysis.TimeListfDOY, UserList[idx_user].Metric[:, 1], 'b+', label='Elevation')
            plt.xlabel('DOY[-]'); plt.ylabel('Azimuth / Elevation [deg]'); plt.legend(); plt.grid()

        if self.Type == 'cov_depth_of_coverage':  # TODO select satellite/constellation
            for i in range(len(SatelliteList)):  # TODO plot ground stations
                plt.scatter(SatelliteList[i].Metric[:, 1], SatelliteList[i].Metric[:, 0], c=SatelliteList[i].Metric[:, 2])
            plt.colorbar(shrink=0.6)
            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()

        if self.Type == 'cov_pass_time':
            plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
            lats, lons = [], []
            time_step = int((Analysis.TimeListMJD[1]-Analysis.TimeListMJD[0])*86400)
            metric = np.zeros(len(UserList))
            for idx_usr in range(len(UserList)):
                valid_value_list = []  # Define and clear
                for idx_sat in range(len(SatelliteList)):
                    for idx_tim in range(1, len(Analysis.TimeListfDOY)):  # Loop over time (ignoring first)
                        if UserList[idx_usr].Metric[idx_tim - 1][idx_sat] and not \
                                UserList[idx_usr].Metric[idx_tim][idx_sat]:  # End pass detected
                                # Compute the length of the pass
                                length_of_pass = 1
                                found_beginning_pass = False
                                while not found_beginning_pass:
                                    if idx_tim - length_of_pass >= 0:
                                        if UserList[idx_usr].Metric[idx_tim - length_of_pass][idx_sat]:
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
                lats.append(UserList[idx_usr].LLA[0] / pi * 180)
                lons.append(UserList[idx_usr].LLA[1] / pi * 180)

            x_new = np.reshape(np.array(lons), (UserList[0].NumLat, UserList[0].NumLon))
            y_new = np.reshape(np.array(lats), (UserList[0].NumLat, UserList[0].NumLon))
            z_new = np.reshape(np.array(metric), (UserList[0].NumLat, UserList[0].NumLon))
            m = Basemap(projection='cyl', lon_0=0)
            im1 = m.pcolormesh(x_new, y_new, z_new, shading='flat', cmap=plt.cm.jet, latlon=True)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            cb = m.colorbar(im1, "right", size="2%", pad="2%")
            cb.set_label(self.Statistic+' Pass Time Interval [s]', fontsize=10)

        if self.Type == 'cov_satellite_highest':
            plt.subplots_adjust(left=.1, right=.9, top=0.99, bottom=0.01)
            metric, lats, lons = [], [], []
            for i in range(len(UserList)):
                if self.Statistic == 'Min':
                    metric.append(np.min(UserList[i].Metric))
                if self.Statistic == 'Mean':
                    metric.append(np.mean(UserList[i].Metric))
                if self.Statistic == 'Max':
                    metric.append(np.max(UserList[i].Metric))
                if self.Statistic == 'Std':
                    metric.append(np.std(UserList[i].Metric))
                if self.Statistic == 'Median':
                    metric.append(np.median(UserList[i].Metric))
                lats.append(UserList[i].LLA[0] / pi * 180)
                lons.append(UserList[i].LLA[1] / pi * 180)
            x_new = np.reshape(np.array(lons), (UserList[0].NumLat, UserList[0].NumLon))
            y_new = np.reshape(np.array(lats), (UserList[0].NumLat, UserList[0].NumLon))
            z_new = np.reshape(np.array(metric), (UserList[0].NumLat, UserList[0].NumLon))
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
            plt.plot(Analysis.TimeListfDOY, data2.x_vel, 'r-', label='x_vel')
            plt.plot(Analysis.TimeListfDOY, data2.y_vel, 'g-', label='y_vel')
            plt.plot(Analysis.TimeListfDOY, data2.z_vel, 'b-', label='z_vel')
            plt.xlabel('DOY[-]'); plt.ylabel('Velocity ECI [m/s]'); plt.legend(); plt.grid()

        plt.savefig('output/'+self.Type+'.png')
        plt.show()

