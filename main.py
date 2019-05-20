# TODO different propagators including SGP4 and HPOP
# TODO analysis COM
# TODO analysis NAV
# TODO analysis OBS

# TODO satellite orbit run from previous run, so to save time
# TODO make available different blocks
# TODO simplify config.xml (cleanup unused headings)
# TODO receiverconstellation by list True/False with commas
# TODO analysis visible satellite for users static, grid and satellite
# TODO analysis one or more users or satellites
# TODO Optimise white space around figures

import os
if os.path.exists('output/main.log'):
    os.remove('output/main.log')
import datetime

import config
import misc_fn
from astropy.time import Time
import logging_svs as ls


def load_configuration():
    sm = config.AppConfig('input/Config.xml')
    sm.load_satellites()
    sm.load_stations()
    sm.load_users()
    sm.load_simulation()
    sm.setup_links()
    return sm  # Configuration is used as state machine


def convert_times(sm):
    run_time_str = Time(sm.time_mjd, format='mjd').iso
    sm.time_gmst = misc_fn.mjd2gmst(sm.time_mjd)  # Determine GMST from MJD
    sm.analysis.times_mjd.append(sm.time_mjd)  # Keep for plotting
    date = datetime.datetime.strptime(run_time_str[:-4], '%Y-%m-%d %H:%M:%S')
    sm.analysis.times_f_doy.append(date.timetuple().tm_yday + date.hour / 24 +
                               date.minute / 60 / 24 + date.second / 3600 / 24)
    sm.time_str = run_time_str[:-4]


def update_satellites(sm):
    # Compute satellite positions in ECF/ECI and compute the links
    # and remember which ones are in view
    for idx_sat, satellite in enumerate(sm.satellites):
        satellite.det_pvt_eci(sm.time_mjd)
        satellite.det_pvt_ecf(sm.time_gmst)
        satellite.num_stat_in_view = 0  # Reset before loop

    # Compute satellite to satellite links
    for idx_sat, satellite in enumerate(sm.satellites):
        if sm.include_sp2sp:  # Only when links needed
            satellite.num_sat_in_view = 0
            for idx_sat2 in range(sm.num_sat):
                if idx_sat != idx_sat2 and sm.sp2sp[idx_sat][idx_sat2].link_in_use:
                    if sm.sp2sp[idx_sat][idx_sat2].compute_link(satellite, sm.satellites[idx_sat2]):
                        satellite.idx_sat_in_view = idx_sat2
                        satellite.num_sat_in_view += 1


def update_stations(sm):
    # Compute ground station positions/velocities in ECI, compute connection to satellite,
    # and remember which ones are in view
    for idx_station, station in enumerate(sm.stations):
        station.num_sat_in_view = 0  # Reset before loop
        station.det_pvt_eci(sm.time_gmst)

        # Compute station to satellite links
        if sm.include_gr2sp:  # Only when links needed
            for idx_sat, satellite in enumerate(sm.satellites):
                if sm.gr2sp[idx_station][idx_sat].link_in_use:  # from receiver constellation
                    sm.gr2sp[idx_station][idx_sat].compute_link(station, satellite)
                    if sm.gr2sp[idx_station][idx_sat].check_masking_station(station):  # Above elevation mask
                        station.idx_sat_in_view[station.num_sat_in_view] = idx_sat
                        station.num_sat_in_view += 1
                        # Compute which stations are in view from this satellite (DOC)
                        satellite.idx_stat_in_view[satellite.num_stat_in_view] = idx_station
                        satellite.num_stat_in_view += 1


def update_users(sm):
    # Compute user positions/velocities in ECI, compute connection to satellite,
    # and remember which ones are in view
    for idx_user, user in enumerate(sm.users):
        user.num_sat_in_view = 0  # Reset before loop
        if user.type == "Static" or user.type == "Grid":
            user.det_pvt_eci(sm.time_gmst)  # Compute position/velocity in ECI
        if user.type == "Spacecraft":
            user.det_pvt_tle(sm.time_gmst, sm.time_mjd)  # Spacecraft position from TLE

        # Compute user to satellite links
        if sm.include_usr2sp:  # Only when links needed
            for idx_sat, satellite in enumerate(sm.satellites):
                if sm.usr2sp[idx_user][idx_sat].link_in_use:  # From Receiver Constellation of User
                    sm.usr2sp[idx_user][idx_sat].compute_link(user, satellite)
                    if sm.usr2sp[idx_user][idx_sat].check_masking_user(user):  # Above elevation mask
                        user.idx_sat_in_view[sm.users[idx_user].num_sat_in_view] = idx_sat
                        user.num_sat_in_view += 1


def main():

    sm = load_configuration()  # load config into sm status machine holds status of sat, station, user and links
    sm.analysis.before_loop(sm)  # Run analysis which is needed before time loop
    ls.logger.info('Finished reading configuration file')

    while round(sm.time_mjd * 86400) < round(sm.stop_time * 86400):  # Loop over simulation time window

        convert_times(sm)  # Convert times in mjd, gmst, string format

        update_satellites(sm)  # Update pvt on satellites and links
        update_stations(sm)  # Update pvt on ground stations and links
        update_users(sm)  # Update pvt on users and links

        sm.analysis.in_loop(sm)  # Run analyses which are needed in time loop

        sm.cnt_epoch += 1
        sm.time_mjd += sm.time_step / 86400.0 # Update time
        ls.logger.info(f'Simulation time: {sm.time_str}, time step: {sm.cnt_epoch}')

    sm.analysis.after_loop(sm)  # Run analysis after time loop
    ls.logger.info(f'Plotting analysis {sm.analysis.type}')


if __name__ == '__main__':
    main()
