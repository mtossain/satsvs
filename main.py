import misc_fn
from astropy.time import Time

import os
try:
    os.remove('main.log')
except:
    pass

import config
import logging
import logging.handlers as handlers

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
logHandler = handlers.RotatingFileHandler('main.log', maxBytes=5*1024*1024)
logHandler.setLevel(logging.INFO)
logHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(logHandler)

conf = config.AppConfig()
conf.LoadSpaceSegment('Config.xml')
conf.LoadGroundSegment('Config.xml')
conf.LoadUserSegment('Config.xml')
conf.SetupGround2Space()
conf.LoadSimulation('Config.xml')

CntEpoch = 0  # timestep
X_sat=[0, 0, 0, 0, 0, 0]  # Temporary state vector

# Run analyses which are needed before time loop
for CntAnalysis in range(len(conf.AnalysisList)):
    conf.AnalysisList[CntAnalysis].RunAnalysisBeforeTimeLoop(conf.NumEpoch, conf.TimeStep, conf.SatelliteList,
                                                             conf.UserList, conf.AnalysisList)

# Loop over simulation time window
logger.info('Starting Simulation')
MJDRequested = conf.StartDateTime
while round(MJDRequested * 86400) < round(conf.StopDateTime * 86400):

    logger.info(['Simulation time MJD:', MJDRequested],)
    GMSTRequested = misc_fn.MJD2GMST(MJDRequested)  # Determine GMST

    if not conf.HPOP:  # Keplerian propagation
        for i in range(conf.NumSat):
            conf.SatelliteList[i].DeterminePosVelECI(MJDRequested)
            conf.SatelliteList[i].DeterminePosVelECF(GMSTRequested)
            conf.SatelliteList[i].NumStationInView = 0  # Reset before loop
    """
    else if (conf.HPOP) {// HPOP propagation
    for (int CntSatellite = 0; CntSatellite < conf.SatelliteList.size(); CntSatellite++) {
    for (int k = CntEpoch * (conf.TimeStep / conf.TimeStepPropagation) + 1; k <= (CntEpoch + 1) * (conf.TimeStep / conf.TimeStepPropagation); k++) {
    for (int i = 0; i < 6; i++)
    X_sat[i] = conf.SatelliteList[CntSatellite].PosVelECI[i];
    HPOP(conf, ParamProp, conf.SatelliteList[CntSatellite].Kepler.SemiMajorAxis, X_sat, CntSatellite,
    conf.SatelliteList[CntSatellite].PosVelECI);
    // Update of the leap seconds and the different IERS parameters (EOP) as well as the different time
    ParamProp.UpdateParam(conf, k);
    ECI2ECF(ParamProp.MJD_TT, ParamProp.MJD_UT1, ParamProp.xp_current, ParamProp.yp_current,
    conf.SatelliteList[CntSatellite].PosVelECI, conf.SatelliteList[CntSatellite].PosVelECF);
    }
    conf.SatelliteList[CntSatellite].NumStationInView = 0; // Reset before loop
    }
    } // End HPOP propagation
    """

    # Compute ground station positions/velocities in ECI, compute connection to satellite,
    # and remember which ones are in view
    for CntGroundStation in range(conf.NumGroundStation):
        conf.GroundStationList[CntGroundStation].NumSatInView = 0  # Reset before loop
        conf.GroundStationList[CntGroundStation].DeterminePosVelECI(GMSTRequested)
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
            conf.UserList[CntUser].DeterminePosVelECI(GMSTRequested)  # Compute position/velocity in ECI
        if conf.UserList[CntUser].Type == "Spacecraft":
            conf.UserList[CntUser].DeterminePosVelTLE(GMSTRequested, MJDRequested)  # Spacecraft position from TLE
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
                #conf.Satellite2SatelliteList[CntSatellite].ComputeLink(conf.SatelliteList[CntSatellite1],conf.SatelliteList[CntSatellite2])
                conf.SatelliteList[CntSatellite1].IdxSatelliteInView = CntSatellite2
                conf.SatelliteList[CntSatellite1].NumSatelliteInView += 1
            CntSatellite += 1

    # Run analyses which are needed in time loop
    for i in range(len(conf.AnalysisList)):
        conf.AnalysisList[i].RunAnalysisInTimeLoop(MJDRequested, CntEpoch, conf.SatelliteList, conf.UserList, conf.User2SatelliteList)

    # Update time
    CntEpoch += 1
    print('Simulation time: '+Time(MJDRequested, format='mjd').iso)
    MJDRequested += conf.TimeStep / 86400.0

# Run analyses which are needed after time loop
for i in range(len(conf.AnalysisList)):
    logger.info(['Plot Analysis:', i])
    conf.AnalysisList[i].RunAnalysisAfterTimeLoop(conf.SatelliteList, conf.UserList)
