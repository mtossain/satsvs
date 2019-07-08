import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
from numpy.linalg import norm
from math import sin, cos, asin, degrees, radians
import xarray as xr
# Project modules
from constants import R_EARTH
from analysis import AnalysisBase, AnalysisObs
import misc_fn
import logging_svs as ls

from multiprocessing import Process, Value, Array, RawArray

class AnalysisObsSwathConical(AnalysisBase, AnalysisObs):

    def __init__(self):
        super().__init__()
        self.polar_view = None
        self.revisit = None
        self.statistic = None
        self.user_pos_ecf = None
        self.user_metric = None
        self.save_output = None
        self.earth_angle_swath = None

    def read_config(self, node):
        if node.find('PolarView') is not None:
            self.polar_view = float(node.find('PolarView').text)
        if node.find('Revisit') is not None:
            self.revisit = misc_fn.str2bool(node.find('Revisit').text)
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text.lower()
        if node.find('SaveOutput') is not None:
            self.save_output = misc_fn.str2bool(node.find('SaveOutput').text)

    def before_loop(self, sm):
        self.det_angles_from_swath_before_loop(sm)
        self.user_pos_ecf = np.zeros((len(sm.users),4))
        self.user_metric = np.zeros((len(sm.users), sm.num_epoch), dtype=np.uint8)
        for idx_user, user in enumerate(sm.users):
            self.user_pos_ecf[idx_user,0:3] = user.pos_ecf
            self.user_pos_ecf[idx_user,3] = norm(self.user_pos_ecf[idx_user,0:3])

    def det_angles_from_swath_before_loop(self, sm):
        for satellite in sm.satellites:
            idx_found = 0
            for idx, constellation in enumerate(sm.constellations):
                if satellite.constellation_id == constellation.constellation_id:
                    idx_found = idx
            const = sm.constellations[idx_found]
            sat_altitude = satellite.kepler.semi_major_axis - R_EARTH
            if const.obs_swath_stop is not None:  # if swath defined by swath length rather than incidence
                satellite.obs_swath_stop = const.obs_swath_stop  #
                satellite.obs_inci_angle_stop = misc_fn.incl_from_swath(
                    const.obs_swath_stop, R_EARTH, sat_altitude)
            else:
                satellite.obs_inci_angle_stop = const.obs_inci_angle_stop
            alfa_critical = asin(R_EARTH / (R_EARTH + sat_altitude))  # If incidence angle shooting off Earth -> error
            if satellite.obs_inci_angle_stop > alfa_critical:
                ls.logger.error(f'Incidence angle stop: {degrees(satellite.obs_inci_angle_stop)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()

    def in_loop(self, sm):
        # Computed by angle distance point and satellite ground point
        # Just 10% faster if done by checking normal euclidean distance
        for satellite in sm.satellites:
            self.det_angles_from_swath_in_loop(satellite)
            self.user_metric[:,sm.cnt_epoch] = \
                misc_fn.check_users_from_nadir(self.user_metric, self.user_pos_ecf, satellite.pos_ecf,
                                               self.earth_angle_swath, sm.cnt_epoch)

    def det_angles_from_swath_in_loop(self, satellite):
        satellite.det_lla()
        r_earth = misc_fn.earth_radius_lat(satellite.lla[0])
        sat_altitude = norm(satellite.pos_ecf) - r_earth
        if satellite.obs_swath_stop is not None:  # if swath defined by swath length rather than incidence
            satellite.obs_inci_angle_stop = misc_fn.incl_from_swath(satellite.obs_swath_stop, r_earth, sat_altitude)
        radius = misc_fn.det_swath_radius(sat_altitude, satellite.obs_inci_angle_stop, r_earth)
        self.earth_angle_swath = misc_fn.earth_angle_beta(radius, r_earth)

    def export2nc(self, sm, file_name):
        user3d_data = np.zeros((len(sm.user_latitudes),len(sm.user_longitudes),sm.num_epoch), dtype=np.uint8)
        for idx_usr, user in enumerate(sm.users):
            idx_lat = np.searchsorted(sm.user_latitudes,degrees(user.lla[0])).flatten()
            idx_lon = np.searchsorted(sm.user_longitudes,degrees(user.lla[1])).flatten()
            user3d_data[int(idx_lat),int(idx_lon),:] = self.user_metric[idx_usr,:]
        da = xr.DataArray(user3d_data,
                          dims=('lat', 'lon', 'time_mjd'),
                          coords={'lat': sm.user_latitudes,
                                  'lon': sm.user_longitudes,
                                  'time_mjd': sm.analysis.times_mjd},
                          name='swath_coverage')
        da.to_netcdf(file_name)


    def after_loop(self, sm):

        if self.save_output=='Numpy':
            np.save('../output/user_cov_swath', self.user_metric)  # Save to numpy array
        if self.save_output=='NetCDF':
            self.export2nc(sm, '../output/user_cov_swath.nc')  # Save to netcdf file

        self.plot_swath_coverage(sm, self.user_metric, self.polar_view)

        if self.revisit:
            self.plot_swath_revisit(sm, self.user_metric, self.statistic, self.polar_view)



class AnalysisObsSwathPushBroom(AnalysisBase, AnalysisObs):

    def __init__(self):
        super().__init__()
        self.polar_view = None
        self.revisit = None
        self.statistic = None
        self.planes = np.zeros((4,3))
        self.user_pos_ecf = None
        self.user_metric = None
        self.save_output = None

    def read_config(self, node):
        if node.find('PolarView') is not None:
            self.polar_view = float(node.find('PolarView').text)
        if node.find('Revisit') is not None:
            self.revisit = misc_fn.str2bool(node.find('Revisit').text)
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text.lower()
        if node.find('SaveOutput') is not None:
            self.save_output = misc_fn.str2bool(node.find('SaveOutput').text)

    def before_loop(self, sm):
        # Get the incidence angles for each of the satelllites
        self.det_angles_from_swath_before_loop(sm)
        self.user_pos_ecf = np.zeros((len(sm.users),3))  # User position in ECF
        self.user_metric = np.zeros((len(sm.users), sm.num_epoch), dtype=np.uint8)  # Range
        self.shared_array = RawArray('i', len(sm.users))
        for idx_user, user in enumerate(sm.users):
            self.user_pos_ecf[idx_user,:] = user.pos_ecf

    def det_angles_from_swath_before_loop(self, sm):
        for satellite in sm.satellites:
            idx_found = 0
            for idx, constellation in enumerate(sm.constellations):
                if satellite.constellation_id == constellation.constellation_id:
                    idx_found = idx
            const = sm.constellations[idx_found]
            sat_altitude = satellite.kepler.semi_major_axis - R_EARTH
            if const.obs_swath_start is not None:  # if swath defined by swath length rather than incidence
                satellite.obs_swath_start = const.obs_swath_start  # Copy over from constellation
                satellite.obs_inci_angle_start = misc_fn.incl_from_swath(
                    const.obs_swath_start, R_EARTH, sat_altitude)
            else:
                satellite.obs_inci_angle_start = const.obs_inci_angle_start
            if const.obs_swath_stop is not None:  # if swath defined by swath length rather than incidence
                satellite.obs_swath_stop = const.obs_swath_stop  # Copy over from constellation
                satellite.obs_inci_angle_stop = misc_fn.incl_from_swath(
                    const.obs_swath_stop, R_EARTH, sat_altitude)
            else:
                satellite.obs_inci_angle_stop = const.obs_inci_angle_stop
            alfa_critical = asin(R_EARTH / (R_EARTH + sat_altitude))  # If incidence angle shooting off Earth -> error
            if np.abs(satellite.obs_inci_angle_start) > alfa_critical:
                ls.logger.error(f'Incidence angle start: {degrees(satellite.obs_inci_angle_start)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()
            if np.abs(satellite.obs_inci_angle_stop) > alfa_critical:
                ls.logger.error(f'Incidence angle stop: {degrees(satellite.obs_inci_angle_stop)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()

    def in_loop(self, sm):

        for satellite in sm.satellites:
            r_earth = self.det_angles_from_swath_in_loop(satellite)
            point_vec1 = misc_fn.rot_vec_vec(-satellite.pos_ecf, np.array(satellite.vel_ecf),
                                             -satellite.obs_inci_angle_start)  # minus for right looking, plus for left
            point_vec2 = misc_fn.rot_vec_vec(-satellite.pos_ecf, np.array(satellite.vel_ecf),
                                             -satellite.obs_inci_angle_stop)  # minus for right looking, plus for left
            intersect, p1b, satellite.p1 = misc_fn.line_sphere_intersect(
                satellite.pos_ecf, satellite.pos_ecf + point_vec1, r_earth, np.zeros(3))
            intersect, p2b, satellite.p2 = misc_fn.line_sphere_intersect(
                satellite.pos_ecf, satellite.pos_ecf + point_vec2, r_earth, np.zeros(3))
            # 4 Planes of pyramid need to be carefully chosen with normal outwards of pyramid
            self.planes[0,:] = misc_fn.plane_normal(satellite.p1, satellite.p2)
            self.planes[1,:] = misc_fn.plane_normal(satellite.p4, satellite.p3)
            self.planes[2,:] = misc_fn.plane_normal(satellite.p2, satellite.p4)
            self.planes[3,:] = misc_fn.plane_normal(satellite.p3, satellite.p1)
            satellite.p3 = satellite.p1.copy()  # Copy for next run, without .copy() python just refers to same list
            satellite.p4 = satellite.p2.copy()
            if sm.cnt_epoch > 0:  # Now valid point 3 and 4
                misc_fn.check_users_in_plane(
                     self.user_pos_ecf, self.planes, self.shared_array)
                # self.user_metric[:,sm.cnt_epoch] = misc_fn.check_users_in_plane(self.user_metric, self.user_pos_ecf,
                #                                                                 self.planes, sm.cnt_epoch)

    def det_angles_from_swath_in_loop(self, satellite):

        satellite.det_lla()
        r_earth = misc_fn.earth_radius_lat(satellite.lla[0])
        sat_altitude = norm(satellite.pos_ecf) - r_earth
        if satellite.obs_swath_start is not None:  # if swath defined by swath length rather than incidence
            satellite.obs_inci_angle_start = misc_fn.incl_from_swath(satellite.obs_swath_start, r_earth, sat_altitude)
        if satellite.obs_swath_stop is not None:  # if swath defined by swath length rather than incidence
            satellite.obs_inci_angle_stop = misc_fn.incl_from_swath(satellite.obs_swath_stop, r_earth, sat_altitude)
        return r_earth

    def export2nc(self, sm, file_name):
        user3d_data = np.zeros((len(sm.user_latitudes),len(sm.user_longitudes),sm.num_epoch), dtype=np.uint8)
        for idx_usr, user in enumerate(sm.users):
            idx_lat = np.searchsorted(sm.user_latitudes,degrees(user.lla[0])).flatten()
            idx_lon = np.searchsorted(sm.user_longitudes,degrees(user.lla[1])).flatten()
            user3d_data[int(idx_lat),int(idx_lon),:] = self.user_metric[idx_usr,:]
        da = xr.DataArray(user3d_data,
                          dims=('lat', 'lon', 'time_mjd'),
                          coords={'lat': sm.user_latitudes,
                                  'lon': sm.user_longitudes,
                                  'time_mjd': sm.analysis.times_mjd},
                          name='swath_coverage')
        da.to_netcdf(file_name)

    def after_loop(self, sm):

        if self.save_output=='Numpy':
            np.save('../output/user_cov_swath', self.user_metric)  # Save to numpy array
        if self.save_output=='NetCDF':
            self.export2nc(sm, '../output/user_cov_swath.nc')  # Save to netcdf file

        self.plot_swath_coverage(sm, self.user_metric, self.polar_view)

        if self.revisit:
            self.plot_swath_revisit(sm, self.user_metric, self.statistic, self.polar_view)
