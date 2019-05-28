import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
from numpy.linalg import norm
# Project modules
from src.constants import R_EARTH
from math import sin, cos, asin, degrees, radians
from src.analysis import AnalysisBase
from src import logging_svs as ls, misc_fn


class AnalysisObsSwathConical(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.ortho_view_latitude = None
        self.revisit = None
        self.statistic = None

    def read_config(self, node):
        if node.find('OrthoViewLatitude') is not None:
            self.ortho_view_latitude = float(node.find('OrthoViewLatitude').text)
        if node.find('Revisit') is not None:
            self.revisit = misc_fn.str2bool(node.find('Revisit').text)
        if node.find('Statistic') is not None:
            self.statistic = node.find('Statistic').text.lower()

    def before_loop(self, sm):
        for satellite in sm.satellites:
            idx_found = 0
            for idx, constellation in enumerate(sm.constellations):
                if satellite.constellation_id == constellation.constellation_id:
                    idx_found = idx
            const = sm.constellations[idx_found]
            sat_altitude = satellite.kepler.semi_major_axis - R_EARTH
            if const.obs_swath_stop is not None:  # if swath defined by swath length rather than incidence
                satellite.obs_incl_angle_stop = misc_fn.incl_from_swath(
                    const.obs_swath_stop, R_EARTH, sat_altitude)
            else:
                satellite.obs_incl_angle_stop = const.obs_incl_angle_stop
            alfa_critical = asin(R_EARTH / (R_EARTH + sat_altitude))  # If incidence angle shooting off Earth -> error
            if satellite.obs_incl_angle_stop > alfa_critical:
                ls.logger.error(f'Inclination angle stop: {degrees(satellite.obs_incl_angle_stop)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)
            user.norm_ecf = norm(user.posvel_ecf[0:3])

    def in_loop(self, sm):
        # Computed by angle distance point and satellite ground point
        # Just 10% faster if done by checking normal euclidean distance
        for satellite in sm.satellites:
            norm_sat = norm(satellite.posvel_ecf[0:3])
            satellite.det_lla()
            sat_altitude = norm_sat - misc_fn.earth_radius_lat(satellite.lla[0])
            radius = misc_fn.det_swath_radius(sat_altitude, satellite.obs_incl_angle_stop)
            earth_angle_swath = radians(misc_fn.earth_angle_beta_deg(radius))
            for user in sm.users:
                angle_user_zenith = misc_fn.angle_two_vectors(user.posvel_ecf[0:3], satellite.posvel_ecf[0:3],
                                                              user.norm_ecf, norm_sat)
                if angle_user_zenith < earth_angle_swath:
                    user.metric[sm.cnt_epoch] = 1  # Within swath

    def after_loop(self, sm):
        plot_points = []
        for user in sm.users:
            if 1 in user.metric:
                plot_points.append([degrees(user.lla[1]), degrees(user.lla[0])])
        x = [row[0] for row in plot_points]
        y = [row[1] for row in plot_points]
        if self.ortho_view_latitude is not None:
            fig = plt.figure(figsize=(7, 7))
            m = Basemap(projection='ortho', lon_0=0, lat_0=self.ortho_view_latitude, resolution='l')
            x, y = m(x,y)
            m.scatter(x, y, 0.05, marker='o', color='r')
        else:
            fig = plt.figure(figsize=(10, 5))
            m = Basemap(projection='cyl', lon_0=0)
            plt.plot(x, y, 'rs', markersize=5, alpha=.3)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.subplots_adjust(left=.1, right=.9, top=0.92, bottom=0.1)
        plt.savefig('../output/'+self.type+'.png')
        plt.show()

        if self.revisit: # If on top revisit is needed
            fig = plt.figure(figsize=(10, 5))
            lats, lons, metric = [], [], []
            for idx_usr, user in enumerate(sm.users):
                valid_value_list = []  # Define and clear
                for idx_tim in range(1, sm.num_epoch):  # Loop over time (ignoring first) and counting backwards
                    if user.metric[idx_tim - 1] == 0 and user.metric[idx_tim] == 1:  # End gap detected
                        length_of_pass = 1  # Compute the length of the pass
                        found_beginning_pass = False
                        while not found_beginning_pass:
                            if idx_tim - length_of_pass >= 0:
                                if user.metric[idx_tim - length_of_pass]==0:
                                    length_of_pass += 1
                                else:
                                    found_beginning_pass = True
                            else:
                                found_beginning_pass = True
                        if abs(idx_tim - length_of_pass)>1:  # Do not count gap between first pass and start simulation
                           valid_value_list.append(length_of_pass * sm.time_step / 3600.0)  # Add gap length for plot

                if len(valid_value_list) == 0:
                    pass  # Do not plot here
                else:
                    if self.statistic == "min":
                        metric.append(np.min(valid_value_list))
                    if self.statistic == "mean":
                        metric.append(np.mean(valid_value_list))
                    if self.statistic == "max":
                        metric.append(np.max(valid_value_list))
                    if self.statistic == "std":
                        metric.append(np.std(valid_value_list))
                    if self.statistic == "median":
                        metric.append(np.median(valid_value_list))
                    lats.append(degrees(sm.users[idx_usr].lla[0]))
                    lons.append(degrees(sm.users[idx_usr].lla[1]))

            m = Basemap(projection='cyl', lon_0=0)
            m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
            m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
            m.drawcoastlines()
            cm = plt.cm.get_cmap('RdYlBu')
            sc = m.scatter(lons,lats, cmap=plt.cm.jet, s=3, c=metric, vmin=0, vmax=np.max(metric))
            cb = m.colorbar(sc, shrink=0.85)
            cb.set_label(self.statistic.capitalize() + ' Revisit Interval [hours]', fontsize=10)
            plt.subplots_adjust(left=.1, right=.9, top=0.9, bottom=0.1)
            plt.savefig(f'../output/{self.type}_revisit.png')
            plt.show()


class AnalysisObsSwathPushBroom(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.ortho_view_latitude = 0
        self.p1 = [0,0,0]  # four corners of the swath
        self.p2 = [0,0,0]  # four corners of the swath
        self.p3 = [0,0,0]  # four corners of the swath
        self.p4 = [0,0,0]  # four corners of the swath

    def read_config(self, node):
        if node.find('OrthoViewLatitude') is not None:
            self.ortho_view_latitude = float(node.find('OrthoViewLatitude').text)

    def before_loop(self, sm):
        # Get the incidence angles for each of the satelllites
        for satellite in sm.satellites:
            idx_found = 0
            for idx, constellation in enumerate(sm.constellations):
                if satellite.constellation_id == constellation.constellation_id:
                    idx_found = idx
            const = sm.constellations[idx_found]
            sat_altitude = satellite.kepler.semi_major_axis - R_EARTH  # Note this is an approximation
            if const.obs_swath_start is not None:  # if swath defined by swath length rather than incidence
                satellite.obs_incl_angle_start = misc_fn.incl_from_swath(
                    const.obs_swath_start, R_EARTH, sat_altitude)
                satellite.obs_incl_angle_stop = misc_fn.incl_from_swath(
                    const.obs_swath_stop, R_EARTH, sat_altitude)
            else:
                satellite.obs_incl_angle_start = const.obs_incl_angle_start
                satellite.obs_incl_angle_stop = const.obs_incl_angle_stop
            alfa_critical = asin(R_EARTH / (R_EARTH + sat_altitude))  # If incidence angle shooting off Earth -> error
            if satellite.obs_incl_angle_start > alfa_critical:
                ls.logger.error(f'Inclination angle start: {degrees(satellite.obs_incl_angle_start)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()
            if satellite.obs_incl_angle_stop > alfa_critical:
                ls.logger.error(f'Inclination angle stop: {degrees(satellite.obs_incl_angle_stop)} ' +
                                f'larger than critical angle {round(degrees(alfa_critical),1)}')
                exit()
        for user in sm.users:
            user.metric = np.zeros(sm.num_epoch)

    def in_loop(self, sm):

        planes = []
        for satellite in sm.satellites:
            sat_pos = np.array(satellite.posvel_ecf[0:3])  # Need np array for easy manipulation
            point_vec1 = misc_fn.rot_vec_vec(-sat_pos, np.array(satellite.posvel_ecf[3:6]),
                                             -satellite.obs_incl_angle_start)  # minus for right looking, plus for left
            point_vec2 = misc_fn.rot_vec_vec(-sat_pos, np.array(satellite.posvel_ecf[3:6]),
                                             -satellite.obs_incl_angle_stop)  # minus for right looking, plus for left
            satellite.det_lla()
            r_earth = misc_fn.earth_radius_lat(satellite.lla[0])  # Determine local radius Earth at latitude
            intersect, p1b, self.p1 = misc_fn.line_sphere_intersect(sat_pos, sat_pos + point_vec1, r_earth, [0, 0, 0])
            intersect, p2b, self.p2 = misc_fn.line_sphere_intersect(sat_pos, sat_pos + point_vec2, r_earth, [0, 0, 0])
            # 4 Planes of pyramid need to be carefully chosen with normal outwards of pyramid
            planes.append(misc_fn.Plane(np.array([0, 0, 0]), np.array(self.p1), np.array(self.p2)))
            planes.append(misc_fn.Plane(np.array([0, 0, 0]), np.array(self.p4), np.array(self.p3)))
            planes.append(misc_fn.Plane(np.array([0, 0, 0]), np.array(self.p2), np.array(self.p4)))
            planes.append(misc_fn.Plane(np.array([0, 0, 0]), np.array(self.p3), np.array(self.p1)))
            self.p3 = self.p1  # Copy for next run
            self.p4 = self.p2  # Copy for next run
            if sm.cnt_epoch > 0:  # Now valid point 3 and 4
                for user in sm.users:
                    if misc_fn.test_point_within_pyramid(np.array(user.posvel_ecf[0:3]), planes):
                        user.metric[sm.cnt_epoch] = 1  # Within swath

    def after_loop(self, sm):
        plot_points = []
        for user in sm.users:
            if 1 in user.metric:
                plot_points.append([degrees(user.lla[1]), degrees(user.lla[0])])
        x = [row[0] for row in plot_points]
        y = [row[1] for row in plot_points]
        if self.ortho_view_latitude > 0:
            fig = plt.figure(figsize=(7, 7))
            m = Basemap(projection='ortho', lon_0=0, lat_0=self.ortho_view_latitude, resolution='l')
            x, y = m(x,y)
            m.scatter(x, y, 0.05, marker='o', color='r')
        else:
            fig = plt.figure(figsize=(10, 5))
            m = Basemap(projection='cyl', lon_0=0)
            plt.plot(x, y, 'rs', markersize=3, alpha=.2)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        m.drawcoastlines()
        plt.subplots_adjust(left=.1, right=.9, top=0.92, bottom=0.1)
        plt.savefig('../output/'+self.type+'.png')
        plt.show()

