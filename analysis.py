import segments
from constants import M_PI
from astropy.time import Time
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
os.environ['PROJ_LIB'] = '/users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
import numpy as np

class Analysis:

    def __init__(self):

        # General analysis parameters
        self.f = ''
        self.Type = 0
        self.Name = 'GroundTrack'
        self.Data = pd.DataFrame()

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
        self.Statistic = 'Mean'  # Depending on analysis it can be Min, Max, Mean, etc.
        self.Direction = 'Hor'  # Depending on analysis it can be Hor, Ver or Pos

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
        self.AtmObstructionHeight  = 0.0  # the height above earth radius to obstruct the link, in m.

        self.PowerInput = 0.0
        self.RecvLineTemp = 0.0  # receiving antenna lines temp
        self.RecvLineLoss = 0.0  #receiving antenna lines losses for noise computation
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
        self.max_link_quality  = 0.0  # to track the maximum non zero in the file

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

        if self.Type == 1:
            # Find the index of the user that is needed
            for i in range(len(UserList)):
                if UserList[i].LLA[0] == self.LatitudeRequested and UserList[i].LLA[1] == self.LongitudeRequested:
                    self.iFndUser = i
                    break

        if self.Type == 2:
            for i in range(len(UserList)):
                UserList[i].Metric = np.zeros(NumEpoch)

        if self.Type == 3:
            # Find the index of the user that is needed
            for i in range(len(UserList)):
                if UserList[i].LLA[0] == self.LatitudeRequested and UserList[i].LLA[1] == self.LongitudeRequested:
                    self.iFndUser = i
                    break

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
                    if UserList[i].LLA[0] == self.LatitudeRequested and UserList[i].LLA[1] == self.LongitudeRequested:
                        self.iFndUser = i
                        break

    def RunAnalysisInTimeLoop(self, RunTime, CntEpoch, SatelliteList, UserList, User2SatelliteList):

        RunTimeStr = Time(RunTime, format='mjd').iso

        if self.Type == 0:
            if self.SatelliteID > 0:  # Only for one satellite
                for i in range(len(SatelliteList)):
                    if SatelliteList[i].ConstellationID == self.ConstellationID and \
                            SatelliteList[i].SatelliteID == self.SatelliteID:
                        SatelliteList[i].DeterminePosVelLLA()
                        self.f.write('{} {} {} {} {} {}\n'.format(RunTimeStr, self.ConstellationID, self.SatelliteID,
                                     SatelliteList[i].LLA[0] / M_PI * 180,
                                     SatelliteList[i].LLA[1] / M_PI * 180,
                                     SatelliteList[i].LLA[2]))
            else:  # Plot the GT for all satellites in the chosen constellation
                for i in range(len(SatelliteList)):
                    if SatelliteList[i].ConstellationID == self.ConstellationID:
                        SatelliteList[i].DeterminePosVelLLA()
                        self.f.write('{} {} {} {} {} {}\n'.format(RunTimeStr, self.ConstellationID, self.SatelliteList[i].SatelliteID,
                                     SatelliteList[i].LLA[0] / M_PI * 180,
                                     SatelliteList[i].LLA[1] / M_PI * 180,
                                     SatelliteList[i].LLA[2]))

        if self.Type == 1:
            self.f.write('{} {} {} {} {}\n'.format(RunTimeStr, self.iFndUser, UserList[self.iFndUser].LLA[0] / M_PI * 180,
                                                      UserList[self.iFndUser].LLA[1] / M_PI * 180, UserList[self.iFndUser].NumSatInView))

        if self.Type == 2:
            # Sum the satellites in view
            for i in range(len(UserList)):
                UserList[i].Metric[CntEpoch] = UserList[i].NumSatInView

        if self.Type == 3:
            for j in range(UserList[self.iFndUser].NumSatInView): # Loop over satellites
                if SatelliteList[UserList[self.iFndUser].IdxSatInView[j]].ConstellationID == self.ConstellationID:
                    self.f.write('{} {} {}\n'.format(RunTimeStr,self.ConstellationID,
                                 SatelliteList[UserList[self.iFndUser].IdxSatInView[j]].SatelliteID))

        if self.Type == 5:
            NumSat = User2SatelliteList[0].NumSat
            if User2SatelliteList[self.iFndUser * NumSat + self.iFndSatellite].Elevation > 0:
                self.f.write('{} {} {} {}\n'.format(RunTimeStr,self.SatelliteID,
                    User2SatelliteList[self.iFndUser * NumSat + self.iFndSatellite].Azimuth / M_PI * 180,
                             User2SatelliteList[self.iFndUser * NumSat + self.iFndSatellite].Elevation / M_PI * 180))

        if self.Type == 6:
            for i in range(len(SatelliteList)):
                SatelliteList[i].DeterminePosVelLLA()
                self.f.write('{} {} {} {}\n'.format(RunTimeStr,SatelliteList[i].LLA[0] / M_PI * 180,
                             SatelliteList[i].LLA[1] / M_PI * 180,SatelliteList[i].NumStationInView))

        if self.Type == 7:
            for i in range(len(UserList)):
                for j in range(UserList[i].NumSatInView): # Loop over satellites
                    if SatelliteList[UserList[i].IdxSatInView[j]].ConstellationID == self.ConstellationID:
                        self.AnalysisMemory3DBool[CntEpoch][i][UserList[i].IdxSatInView[j]] = True

        if self.Type == 8:
            # Determine whether X or more satellites are in view
            for i in range(len(UserList)):
                CntSat = 0
                for j in range(UserList[i].NumSatInView): # Loop over satellites
                    if SatelliteList[UserList[i].IdxSatInView[j]].ConstellationID == self.ConstellationID:
                        CntSat += 1
                if CntSat >= self.RequiredNumberSatellites:
                    self.AnalysisMemory2DBool[CntEpoch][i] = True

    def RunAnalysisAfterTimeLoop(self):
        self.f.close()

    def Plot(self, UserList):
        if self.Type == 0:
            data = pd.read_csv('analysis_0.txt', delim_whitespace=True, header=None)
            fig = plt.figure(figsize=(12,8))
            m = Basemap(projection='cyl',lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            plt.plot(data.iloc[:, 5], data.iloc[:, 4],'r.')
            plt.savefig('analysis_0.png')
            plt.show()

        if self.Type == 1:
            df = pd.read_csv('analysis_1.txt', delim_whitespace=True, header=None)
            df["Date"] = pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1])
            df['fDOY'] = df['Date'].dt.dayofyear + df['Date'].dt.hour/24+df['Date'].dt.minute/60/24+df['Date'].dt.second/3600/24
            fig = plt.figure(figsize=(12,8))
            plt.plot(df['fDOY'], df.iloc[:, 5], 'r-')
            plt.xlabel('DOY[-]'); plt.ylabel('Number of satellites in view'); plt.grid()
            plt.savefig('analysis_1.png')
            plt.show()

        if self.Type == 2:
            Metric,Lats,Lons = [],[],[]
            for i in range(len(UserList)):
                if self.Statistic == 'Min':
                    Metric.append(np.min(UserList[i].Metric))
                if self.Statistic == 'Mean':
                    Metric.append(np.mean(UserList[i].Metric))
                if self.Statistic == 'Max':
                    Metric.append(np.max(UserList[i].Metric))
                if self.Statistic == 'Std':
                    Metric.append(np.std(UserList[i].Metric))
                if self.Statistic == 'Median':
                    Metric.append(np.median(UserList[i].Metric))
                Lats.append(UserList[i].LLA[0]/M_PI*180)
                Lons.append(UserList[i].LLA[1]/M_PI*180)
            Xnew = np.reshape(np.array(Lons), (19,37))
            Ynew = np.reshape(np.array(Lats), (19,37))
            Znew = np.reshape(np.array(Metric), (19,37))
            fig = plt.figure(figsize=(12,8))
            m = Basemap(projection='cyl', lon_0=0)
            im1 = m.pcolormesh(Xnew,Ynew,Znew, shading='flat', cmap=plt.cm.jet, latlon=True)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            cb = m.colorbar(im1, "right", size="2%", pad="2%")
            cb.set_label(self.Statistic+' Number of satellites in view', fontsize=10)
            plt.savefig('analysis_2.png')
            plt.show()


    def ComputeStatistics(self):
        pass