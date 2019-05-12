# TODO compare against C# reference implementation
# TODO make available different block
# TODO write user manual in GIT Readme
# TODO different propagators including SGP4 and HPOP
# TODO satellite orbit run from previous run, so to save time
# TODO S2G and S2S only when needed with flag...
# TODO analysis 5-10
# TODO analysis 20
# TODO analysis navigation
# TODO analysis earth observation
# TODO refactor names of variables and classes
import os
try:
    os.remove('output/main.log')
except:
    pass

import config
import misc_fn
from astropy.time import Time
import logging_svs as ls
ls.logger.info('Read configuration file')
from analysis import Analysis
import datetime


def main():

    # Loading configuration
    conf = config.AppConfig()
    conf.LoadSpaceSegment('input/Config.xml')
    conf.LoadGroundSegment('input/Config.xml')
    conf.LoadUserSegment('input/Config.xml')
    conf.LoadSimulation('input/Config.xml')
    conf.SetupGround2Space()

    # Run analyses which are needed before time loop
    for CntAnalysis in range(len(conf.AnalysisList)):
        conf.AnalysisList[CntAnalysis].RunAnalysisBeforeTimeLoop(conf.NumEpoch, conf.TimeStep, conf.SatelliteList,
                                                                 conf.UserList, conf.AnalysisList)
    # Loop over simulation time window
    mjd_requested = conf.StartDateTime
    cnt_epoch = 0  # count the time_steps
    while round(mjd_requested * 86400) < round(conf.StopDateTime * 86400):

        run_time_str = Time(mjd_requested, format='mjd').iso
        gmst_requested = misc_fn.MJD2GMST(mjd_requested)  # Determine GMST
        Analysis.TimeListISO.append(run_time_str)
        Analysis.TimeListMJD.append(mjd_requested)
        date = datetime.datetime.strptime(run_time_str[:-4], '%Y-%m-%d %H:%M:%S')
        Analysis.TimeListfDOY.append(date.timetuple().tm_yday + date.hour/24+date.minute/60/24+date.second/3600/24)
        ls.logger.info(['Sim Time:', run_time_str])

        for i in range(conf.NumSat):
            conf.SatelliteList[i].DeterminePosVelECI(mjd_requested)
            conf.SatelliteList[i].DeterminePosVelECF(gmst_requested)
            conf.SatelliteList[i].NumStationInView = 0  # Reset before loop

        # Compute ground station positions/velocities in ECI, compute connection to satellite,
        # and remember which ones are in view
        for CntGroundStation in range(conf.NumGroundStation):
            conf.GroundStationList[CntGroundStation].NumSatInView = 0  # Reset before loop
            conf.GroundStationList[CntGroundStation].DeterminePosVelECI(gmst_requested)
            for CntSatellite in range(conf.NumSat):
                if conf.GroundStation2SatelliteList[CntGroundStation * conf.NumSat + CntSatellite].LinkInUse:  # From Receiver Constellation of Ground Station
                    conf.GroundStation2SatelliteList[CntGroundStation * conf.NumSat + CntSatellite].\
                        ComputeLinkGroundStation(conf.GroundStationList[CntGroundStation], conf.SatelliteList[CntSatellite])
                    if conf.GroundStation2SatelliteList[CntGroundStation * conf.NumSat + CntSatellite].\
                            CheckMasking(conf.GroundStationList[CntGroundStation]):  # Above elevation mask
                        conf.GroundStationList[CntGroundStation].IdxSatInView[conf.GroundStationList[CntGroundStation].NumSatInView] = CntSatellite
                        conf.GroundStationList[CntGroundStation].NumSatInView += 1
                        # Compute which stations are in view from this satellite (DOC)
                        conf.SatelliteList[CntSatellite].IdxStationInView[conf.SatelliteList[CntSatellite].NumStationInView] = CntGroundStation
                        conf.SatelliteList[CntSatellite].NumStationInView += 1

        # Compute user positions/velocities in ECI, compute connection to satellite, and remember which ones are in view
        for CntUser in range(conf.NumUser):
            conf.UserList[CntUser].NumSatInView = 0  # Reset before loop
            if conf.UserList[CntUser].Type == "Static" or conf.UserList[CntUser].Type == "Grid":
                conf.UserList[CntUser].DeterminePosVelECI(gmst_requested)  # Compute position/velocity in ECI
            if conf.UserList[CntUser].Type == "Spacecraft":
                conf.UserList[CntUser].DeterminePosVelTLE(gmst_requested, mjd_requested)  # Spacecraft position from TLE
            for CntSatellite in range(conf.NumSat):
                if conf.User2SatelliteList[CntUser * conf.NumSat + CntSatellite].LinkInUse:  # From Receiver Constellation of User
                    conf.User2SatelliteList[CntUser * conf.NumSat + CntSatellite].\
                        ComputeLinkUser(conf.UserList[CntUser], conf.SatelliteList[CntSatellite])
                    if conf.User2SatelliteList[CntUser * conf.NumSat + CntSatellite].CheckMasking(conf.UserList[CntUser]):  # Above elevation mask
                        conf.UserList[CntUser].IdxSatInView[conf.UserList[CntUser].NumSatInView] = CntSatellite
                        conf.UserList[CntUser].NumSatInView += 1

        # Compute satellite to satellite links
        CntSatellite = 0
        for CntSatellite1 in range(conf.NumSat):
            conf.SatelliteList[CntSatellite1].NumSatelliteInView = 0
            for CntSatellite2 in range(conf.NumSat):
                if conf.Satellite2SatelliteList[CntSatellite].LinkInUse:  # From Receiver Constellation of Spacecraft TBD
                    # computing links. If Satellite1 == Satellite2, ComputeLink return 0
                    #conf.Satellite2SatelliteList[CntSatellite].ComputeLink(conf.SatelliteList[CntSatellite1],conf.SatelliteList[CntSatellite2]) # TODO Check space to space link
                    conf.SatelliteList[CntSatellite1].IdxSatelliteInView = CntSatellite2
                    conf.SatelliteList[CntSatellite1].NumSatelliteInView += 1
                CntSatellite += 1

        # Run analyses which are needed in time loop
        for i in range(len(conf.AnalysisList)):
            conf.AnalysisList[i].RunAnalysisInTimeLoop(mjd_requested, cnt_epoch, conf.SatelliteList, conf.UserList,
                                                       conf.User2SatelliteList)

        # Update time
        cnt_epoch += 1
        mjd_requested += conf.TimeStep / 86400.0

    # Run analyses which are needed after time loop
    for i in range(len(conf.AnalysisList)):
        ls.logger.info(['Plot Analysis:', conf.AnalysisList[i].Type])
        conf.AnalysisList[i].RunAnalysisAfterTimeLoop(conf.SatelliteList, conf.UserList)


if __name__ == '__main__':
    main()