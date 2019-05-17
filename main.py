# TODO make available different block
# TODO write user manual in GIT Readme
# TODO different propagators including SGP4 and HPOP
# TODO satellite orbit run from previous run, so to save time
# TODO S2G and S2S only when needed with flag...
# TODO analysis 20 series
# TODO analysis navigation
# TODO analysis earth observation
# TODO refactor names of variables and classes
# TODO simplify config.xml (cleanup unused headings)
# TODO receiverconstellation by list True/False with commas
# TODO analysis visible satellite for users static, grid and satellite
import os
if os.path.exists('output/main.log'):
    os.remove('output/main.log')
import datetime

import config
import misc_fn
from astropy.time import Time
import logging_svs as ls


def load_configuration():
    sm = config.AppConfig('input/Config.xml' )
    sm.load_satellites()
    sm.load_stations()
    sm.load_users()
    sm.load_simulation()
    sm.setup_links()
    return sm  # Configuration is used as state machine


def convert_times(sm):
    run_time_str = Time(sm.time_mjd, format='mjd').iso
    sm.time_gmst = misc_fn.MJD2GMST(sm.time_mjd)  # Determine GMST from MJD
    sm.analysis.times_mjd.append(sm.time_mjd)  # Keep for plotting
    date = datetime.datetime.strptime(run_time_str[:-4], '%Y-%m-%d %H:%M:%S')
    sm.analysis.times_fDOY.append(date.timetuple().tm_yday + date.hour / 24 +
                               date.minute / 60 / 24 + date.second / 3600 / 24)
    sm.time_str = run_time_str[:-4]


def update_satellites(sm):
    CntSatellite = 0
    for satellite in sm.satellites:
        satellite.DeterminePosVelECI(sm.time_mjd)
        satellite.DeterminePosVelECF(sm.time_gmst)
        satellite.NumStationInView = 0  # Reset before loop
        # Compute satellite to satellite links # TODO Check space to space link
        satellite.NumSatelliteInView = 0
        # for CntSatellite2 in range(sm.NumSat):
        #     if sm.sp2sp[CntSatellite].LinkInUse:  # From Receiver Constellation of Spacecraft TBD
        #         sm.sp2sp[CntSatellite].ComputeLink(satellite, sm.satellites[CntSatellite2])
        #         satellite.IdxSatelliteInView = CntSatellite2
        #         satellite.NumSatelliteInView += 1
        #         CntSatellite += 1

def update_stations(sm):
    # Compute ground station positions/velocities in ECI, compute connection to satellite,
    # and remember which ones are in view
    for idx_station,station in enumerate(sm.stations):
        station.NumSatInView = 0  # Reset before loop
        station.DeterminePosVelECI(sm.time_gmst)
        for idx_sat,satellite in enumerate(sm.satellites):
            if sm.gr2sp[idx_station * sm.NumSat + idx_sat].LinkInUse:  # from receiver constellation
                sm.gr2sp[idx_station * sm.NumSat + idx_sat].ComputeLinkGroundStation(station, satellite)
                if sm.gr2sp[idx_station * sm.NumSat + idx_sat].CheckMaskingStation(station):  # Above elevation mask
                    station.IdxSatInView[station.NumSatInView] = idx_sat
                    station.NumSatInView += 1
                    # Compute which stations are in view from this satellite (DOC)
                    satellite.IdxStationInView[satellite.NumStationInView] = idx_station
                    satellite.NumStationInView += 1


def update_users(sm):
    # Compute user positions/velocities in ECI, compute connection to satellite,
    # and remember which ones are in view
    for idx_user,user in enumerate(sm.users):
        user.NumSatInView = 0  # Reset before loop
        if user.Type == "Static" or user.Type == "Grid":
            user.DeterminePosVelECI(sm.time_gmst)  # Compute position/velocity in ECI
        if user.Type == "Spacecraft":
            user.DeterminePosVelTLE(sm.time_gmst, sm.time_mjd)  # Spacecraft position from TLE
        for idx_sat, satellite in enumerate(sm.satellites):
            idx_usr2sat = idx_user * sm.NumSat + idx_sat
            if sm.usr2sp[idx_usr2sat].LinkInUse:  # From Receiver Constellation of User
                sm.usr2sp[idx_usr2sat].ComputeLinkUser(user, satellite)
                if sm.usr2sp[idx_usr2sat].CheckMaskingUser(user):  # Above elevation mask
                    user.IdxSatInView[sm.users[idx_user].NumSatInView] = idx_sat
                    user.NumSatInView += 1


def main():

    sm = load_configuration()  # load config into sm status machine holds status of sat, station, user and links
    ls.logger.info('Read configuration file')

    sm.analysis.before_loop(sm)  # Run analysis which is needed before time loop

    while round(sm.time_mjd * 86400) < round(sm.StopDateTime * 86400):  # Loop over simulation time window

        convert_times(sm)  # Convert times in mjd, gmst, string format

        update_satellites(sm)  # Update pvt on satellites and links

        update_stations(sm)  # Update pvt on ground stations and links

        update_users(sm)  # Update pvt on users and links

        sm.analysis.in_loop(sm)  # Run analyses which are needed in time loop

        sm.cnt_epoch += 1
        sm.time_mjd += sm.TimeStep / 86400.0 # Update time
        ls.logger.info(['Sim Time:', sm.time_str, 'Time Step:', str(sm.cnt_epoch)])

    sm.analysis.after_loop(sm)  # Run analysis after time loop
    ls.logger.info(['Plot Analysis:', sm.analysis.Type])


if __name__ == '__main__':
    main()
