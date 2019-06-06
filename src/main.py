# Import python modules
import os
if os.path.exists('../output/main.log'):
    os.remove('../output/main.log')
import datetime
import numpy as np
# Import project modules
import config
import logging_svs as ls
import misc_fn
from astropy.time import Time


def main():

    sm = load_configuration()  # Load config into sm status machine holds status of sat, station, user and links
    sm.analysis.before_loop(sm)  # Run analysis which is needed before time loop
    ls.logger.info('Finished reading configuration file')

    clear_load_orbit_file(sm)  # Clear or load previous orbit file

    for sm.cnt_epoch in range(sm.num_epoch):  # Loop over simulation time window

        convert_times(sm)  # Convert times from mjd to gmst, fdoy and string format
        ls.logger.info(f'Simulation time: {sm.time_str}, time step: {sm.cnt_epoch}')

        update_satellites(sm)  # Update pvt on satellites and links
        update_stations(sm)  # Update pvt on ground stations and links
        update_users(sm)  # Update pvt on users and links

        sm.analysis.in_loop(sm)  # Run analyses which are needed in time loop

        sm.time_mjd += sm.time_step / 86400.0  # Update time

    ls.logger.info(f'Plotting analysis {sm.analysis.type}')
    sm.analysis.after_loop(sm)  # Run analysis after time loop


def load_configuration():
    sm = config.AppConfig('../input/Config.xml')
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
    sm.time_datetime = date


def update_satellites(sm):
    # Compute satellite positions in ECF/ECI and compute the links
    # and remember which ones are in view
    if sm.orbits_from_previous_run:  # Read the ECI posvel from all satellites for one epoch from file
        for idx_sat, satellite in enumerate(sm.satellites):
            satellite.pos_eci = sm.data_orbits[sm.cnt_epoch * sm.num_sat + idx_sat, 0:3]
            satellite.vel_eci = sm.data_orbits[sm.cnt_epoch * sm.num_sat + idx_sat, 3:6]
            satellite.det_posvel_ecf(sm.time_gmst)
            satellite.idx_stat_in_view = []  # Reset before loop
    else:
        for idx_sat, satellite in enumerate(sm.satellites):
            if sm.orbit_propagator == 'Keplerian':
                satellite.det_posvel_eci_keplerian(sm.time_mjd)
            if sm.orbit_propagator == 'SGP4':
                satellite.det_posvel_eci_sgp4(sm.time_datetime)
            satellite.det_posvel_ecf(sm.time_gmst)
            satellite.idx_stat_in_view = []  # Reset before loop
        write_posvel_satellites(sm)

    # Compute satellite to satellite links
    for idx_sat, satellite in enumerate(sm.satellites):
        if sm.include_sp2sp:  # Only when links needed
            satellite.idx_sat_in_view = []  # Reset before loop
            for idx_sat2, satellite2 in enumerate(sm.satellites):
                if sm.sp2sp[idx_sat][idx_sat2].link_in_use:
                    sm.sp2sp[idx_sat][idx_sat2].compute_link(satellite, sm.satellites[idx_sat2])
                    # Check if above elevation mask and not through Earth
                    if sm.sp2sp[idx_sat][idx_sat2].check_masking(satellite, satellite2):
                        satellite.idx_sat_in_view.append(idx_sat2)


def update_stations(sm):
    # Compute ground station positions/velocities in ECI, compute connection to satellite,
    # and remember which ones are in view
    if sm.include_gr2sp:  # Only when links needed
        for idx_station, station in enumerate(sm.stations):
            station.idx_sat_in_view = []  # Reset before loop
            station.det_posvel_eci(sm.time_gmst)

            # Compute station to satellite links
            for idx_sat, satellite in enumerate(sm.satellites):
                if sm.gr2sp[idx_station][idx_sat].link_in_use:  # from receiver constellation
                    sm.gr2sp[idx_station][idx_sat].compute_link(station, satellite)
                    if sm.gr2sp[idx_station][idx_sat].check_masking(station):  # Above elevation mask
                        station.idx_sat_in_view.append(idx_sat)
                        # Compute which stations are in view from this satellite (DOC)
                        satellite.idx_stat_in_view.append(idx_station)


def update_users(sm):
    # Compute user positions/velocities in ECI, compute connection to satellite,
    # and remember which ones are in view
    if sm.include_usr2sp:  # Only when links needed
        for idx_user, user in enumerate(sm.users):
            user.idx_sat_in_view = []  # Reset before loop
            if user.type == "Static" or user.type == "Grid":
                user.det_posvel_eci(sm.time_gmst)  # Compute position/velocity in ECI
            if user.type == "Spacecraft":
                user.det_posvel_tle(sm.time_gmst, sm.time_mjd)  # Spacecraft position from TLE

            # Compute user to satellite links
            for idx_sat, satellite in enumerate(sm.satellites):
                if sm.usr2sp[idx_user][idx_sat].link_in_use:  # From Receiver Constellation of User
                    sm.usr2sp[idx_user][idx_sat].compute_link(user, satellite)
                    if sm.usr2sp[idx_user][idx_sat].check_masking(user):  # Above elevation mask
                        user.idx_sat_in_view.append(idx_sat)


def write_posvel_satellites(sm):  # Write the orbits from file
    pass
    for idx_sat, satellite in enumerate(sm.satellites):
        with open('../output/orbits_internal.txt', 'a') as f:
            f.write("%13.6f,%13.6f,%13.6f,%13.6f,%13.6f,%13.6f\n"
                    % (satellite.pos_eci[0], satellite.pos_eci[1], satellite.pos_eci[2],
                       satellite.vel_eci[0], satellite.vel_eci[1], satellite.vel_eci[1]))


def clear_load_orbit_file(sm):  # Clear or load previous orbit file
    if sm.orbits_from_previous_run:
        sm.data_orbits = np.genfromtxt('../output/orbits_internal.txt', delimiter=',')
    else:
        if os.path.exists('../output/orbits_internal.txt'):
            os.remove('../output/orbits_internal.txt')


if __name__ == '__main__':
    main()

# TODO SGP4 for Keplerian
# TODO analysis COM
# TODO analysis NAV

