
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

    TimeListISO = []
    TimeListMJD = []
    TimeListfDOY = []

    def __init__(self):

        # General analysis parameters
        self.Type = ''
        self.TimeListISO = []
        self.TimeListMJD = []
        self.TimeListfDOY = []

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

        if self.Type == 0:
            for i in range(len(SatelliteList)):
                SatelliteList[i].Metric = np.zeros((NumEpoch,2))

        if self.Type == 1:
            # Find the index of the user that is needed
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros(NumEpoch)

        if self.Type == 2:
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros(NumEpoch)

        if self.Type == 3:
            # Find the index of the user that is needed
            for i in range(len(UserList)):
                UserList[i].Metric = np.ones((NumEpoch, len(SatelliteList)))*999999

        if self.Type == 4:
            # Find the index of the satellite that is needed
            for i in range(len(SatelliteList)):
                if SatelliteList[i].ConstellationID == self.ConstellationID and \
                        SatelliteList[i].SatelliteID == self.SatelliteID:
                    self.iFndSatellite = i
                    break

        if self.Type == 5:
            # Find the index of the satellite that is needed
            for i in range(len(SatelliteList)):
                if SatelliteList[i].ConstellationID == self.ConstellationID and \
                        SatelliteList[i].SatelliteID == self.SatelliteID:
                    self.iFndSatellite = i
                    break
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros((NumEpoch, 2))

        if self.Type == 6:
            for i in range(len(SatelliteList)):
                SatelliteList[i].Metric = np.zeros((NumEpoch,3))

        if self.Type == 7:
            for i in range(len(UserList)):
                UserList[i].Metric = np.full((NumEpoch, len(SatelliteList)), False, dtype=bool)

        if self.Type == 8:
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros(NumEpoch)

        if self.Type == 'cov_satellite_pvt':
            try:
                os.remove('output/orbits.txt')
            except:
                pass

    def RunAnalysisInTimeLoop(self, RunTime, CntEpoch, SatelliteList, UserList, User2SatelliteList):

        if self.Type == 0:
            if self.SatelliteID > 0:  # Only for one satellite
                for idx in range(len(SatelliteList)):
                    if SatelliteList[idx].ConstellationID == self.ConstellationID and \
                            SatelliteList[idx].SatelliteID == self.SatelliteID:
                        SatelliteList[idx].DeterminePosVelLLA()
                        SatelliteList[idx].Metric[CntEpoch, 0] = SatelliteList[idx].LLA[0] / pi * 180
                        SatelliteList[idx].Metric[CntEpoch, 1] = SatelliteList[idx].LLA[1] / pi * 180
            else:  # Plot the GT for all satellites in the chosen constellation
                for idx in range(len(SatelliteList)):
                    if SatelliteList[idx].ConstellationID == self.ConstellationID:
                        SatelliteList[idx].DeterminePosVelLLA()
                        SatelliteList[idx].Metric[CntEpoch, 0] = SatelliteList[idx].LLA[0] / pi * 180
                        SatelliteList[idx].Metric[CntEpoch, 1] = SatelliteList[idx].LLA[1] / pi * 180

        if self.Type == 1:
            for idx in range(len(UserList)):
                UserList[idx].Metric[CntEpoch] = UserList[idx].NumSatInView

        if self.Type == 2:
            for idx in range(len(UserList)):
                UserList[idx].Metric[CntEpoch] = UserList[idx].NumSatInView

        if self.Type == 3:
            for idx in range(len(UserList[0].IdxSatInView)):
                if UserList[0].IdxSatInView[idx] != 999999:
                    UserList[0].Metric[CntEpoch, idx] = SatelliteList[UserList[0].IdxSatInView[idx]].SatelliteID

        if self.Type == 5:
            num_sat = User2SatelliteList[0].NumSat
            for idx in range(len(UserList)):
                if User2SatelliteList[idx * num_sat + self.iFndSatellite].Elevation > 0:
                    UserList[idx].Metric[CntEpoch, 0] = User2SatelliteList[idx * num_sat + self.iFndSatellite].Azimuth / pi * 180
                    UserList[idx].Metric[CntEpoch, 1] = User2SatelliteList[idx * num_sat + self.iFndSatellite].Elevation / pi * 180

        if self.Type == 6:
            for idx in range(len(SatelliteList)):
                SatelliteList[idx].DeterminePosVelLLA()
                SatelliteList[idx].Metric[CntEpoch, 0] = SatelliteList[idx].LLA[0] / pi * 180
                SatelliteList[idx].Metric[CntEpoch, 1] = SatelliteList[idx].LLA[1] / pi * 180
                SatelliteList[idx].Metric[CntEpoch, 2] = SatelliteList[idx].NumStationInView

        if self.Type == 7:
            for idx in range(len(UserList)):
                for j in range(UserList[idx].NumSatInView):
                    if SatelliteList[UserList[idx].IdxSatInView[j]].ConstellationID == self.ConstellationID:
                        UserList[idx].Metric[CntEpoch, UserList[idx].IdxSatInView[j]] = True

        if self.Type == 8: # TODO ERROR !!!! there seems to be a bug in the user to satellite elevation value, it has to be checked
            for idx_user in range(len(UserList)):
                best_satellite_value = -1
                for idx_sat in range(UserList[idx_user].NumSatInView):
                    if SatelliteList[UserList[idx_user].IdxSatInView[idx_sat]].ConstellationID == self.ConstellationID:
                        elevation = User2SatelliteList[idx_user * len(SatelliteList) + UserList[idx_user].IdxSatInView[idx_sat]].Elevation / pi * 180
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

        fig = plt.figure(figsize=(12, 8))
        #plt.tight_layout()
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)

        if self.Type == 0:
            if self.SatelliteID > 0:  # Only for one satellite
                for i in range(len(SatelliteList)):
                    if SatelliteList[i].ConstellationID == self.ConstellationID and \
                            SatelliteList[i].SatelliteID == self.SatelliteID:
                        y,x = SatelliteList[i].Metric[:,0], SatelliteList[i].Metric[:,1]
            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            plt.plot(x, y, 'r.')

        if self.Type == 1:
            for i in range(len(UserList)):  # TODO check multiple users
                plt.plot(self.TimeListfDOY, UserList[i].Metric, 'r-')
            plt.xlabel('DOY[-]'); plt.ylabel('Number of satellites in view'); plt.grid()

        if self.Type == 2:
            metric,lats,lons = [],[],[]
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

        if self.Type == 3:
            plt.plot(self.TimeListfDOY, UserList[0].Metric, 'r+')
            plt.ylim((.5, len(SatelliteList)+1))
            plt.xlabel('DOY[-]'); plt.ylabel('Number of satellites in view'); plt.grid()

        if self.Type == 4:
            SatelliteList[self.iFndSatellite].DeterminePosVelLLA()
            Contour = misc_fn.SatGrndVis(SatelliteList[self.iFndSatellite].LLA, self.ElevationMask)
            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            plt.plot(Contour[:,1] / pi * 180, Contour[:, 0] / pi * 180, 'r.')

        if self.Type == 5:  # TODO check multiple users
            for i in range(len(UserList)):
                plt.plot(self.TimeListfDOY, UserList[i].Metric[:, 0], 'r+',label='Azimuth')
                plt.plot(self.TimeListfDOY, UserList[i].Metric[:, 1], 'b+',label='Elevation')
            plt.xlabel('DOY[-]'); plt.ylabel('Azimuth / Elevation [deg]'); plt.legend(); plt.grid()

        if self.Type == 6:
            for i in range(len(SatelliteList)):
                plt.scatter(SatelliteList[i].Metric[:, 1], SatelliteList[i].Metric[:, 0], c=SatelliteList[i].Metric[:, 2])
            plt.colorbar(shrink=0.6)
            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()

        if self.Type == 7:

            lats, lons = [], []
            time_step = int((self.TimeListMJD[1]-self.TimeListMJD[0])*86400)
            metric = np.zeros(len(UserList))

            for idx_usr in range(len(UserList)):
                valid_value_list = []  # Define and clear
                for idx_sat in range(len(SatelliteList)):
                    for idx_tim in range(1, len(self.TimeListfDOY)):  # Loop over time (ignoring first)
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

        if self.Type == 8:
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
            plt.plot(Analysis.TimeListfDOY, data2.x_vel, 'r-')
            plt.xlabel('DOY[-]'); plt.ylabel('X value velocity ECI [m/s]'); plt.grid()

        plt.savefig('output/'+self.Type+'.png')
        plt.show()

