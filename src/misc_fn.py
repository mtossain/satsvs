import math
from math import sin, cos, atan2, atan, fabs, acos
import numpy as np
from astropy.coordinates import EarthLocation
from math import tan, sqrt, asin, degrees, radians
from numpy import dot, arccos, cross, sin, cos
from numpy.linalg import norm
from numba import jit, float64
from scipy import interpolate
# Modules from project
from constants import R_EARTH, PI, GM_EARTH


def plane_normal(a,b):  # assume here point c is [0,0,0]
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c

# Compute Modified Julian Date from YYYY and Day Of Year pair
# Time functions from SP3 library B.Remondi.
#
# @param YYYY Year (year)
# @param DOY Day of year (days)
# @return Modified Julian Date
def yyyy_doy2mjd(yyyy, doy):

    mjd_jan1_1901 = 15385
    days_per_second = 0.00001157407407407407

    y_day = int(math.floor(doy))
    f_doy = doy - y_day
    full_seconds = f_doy * 86400.0
    hours = full_seconds / 3600.0
    hour = int(math.floor(hours))
    rest = full_seconds - hour * 3600
    minutes = rest / 60.0
    minute = int(math.floor(minutes))
    second = full_seconds - hour * 3600 - minute * 60

    years_into_election = (yyyy - 1) % 4
    elections = int((yyyy - 1901) / 4)

    pf_mjd = (hour * 3600 + minute * 60 + second) * days_per_second
    p_mjd = mjd_jan1_1901 + elections * 1461 + years_into_election * 365 + y_day - 1

    return pf_mjd + p_mjd


# Compute Greenwich Mean Sidereal Time from Modified Julian Date
# Ref. Time functions from SP3 library B.Remondi.
#
# @param MJD_Requested Modified Julian Date (days/double)
# @return Greenwich Mean Sidereal Time (radians/double) [0 to 2PI]
def mjd2gmst(mjd_requested):

    jd = mjd_requested + 2400000.5

    # centuries elapsed since 2000,1,1.5
    tu = (jd - 2451545) / 36525

    # gmst in time-seconds at 0 ut
    theta_n = 24110.54841 + (8640184.812866 * tu)+(0.093104 * tu * tu)-(6.2e-6 * tu * tu * tu)

    # correction for other time than 0 ut modulo rotations
    theta_0 = 43200.0 + (3155760000.0 * tu)
    theta_n = theta_n + (theta_0 % 86400)

    # gmst in radians
    theta_n = ((theta_n / 43200) * PI) % (2 * PI)
    if theta_n < 0:
        theta_n = theta_n + (2 * PI)

    return theta_n


# Convert from latitude, longitude and height to ECF cartesian coordinates, taking
# into account the Earth flattening
#
# Ref. Understanding GPS: Principles and Applications,
# Elliott D. Kaplan, Editor, Artech House Publishers, Boston 1996.
#
# @param LLA Latitude rad/[-PI/2,PI/2] positive N, Longitude rad/[-PI,PI] positive E, height above ellipsoid
# @param XYZ ECEF XYZ coordinates in m
@jit(nopython=True)
def lla2xyz(lla):

    xyz = np.zeros(3)
    
    a = 6378137.0000
    b = 6356752.3142
    e = np.sqrt(1 - np.power((b / a), 2))
    
    sin_phi = np.sin(lla[0])
    cos_phi = np.cos(lla[0])
    cos_lam = np.cos(lla[1])
    sin_lam = np.sin(lla[1])
    tan2phi = np.power((tan(lla[0])), 2.0)
    tmp = 1.0 - e*e
    tmp_den = np.sqrt(1.0 + tmp * tan2phi)
    
    xyz[0] = (a * cos_lam) / tmp_den + lla[2] * cos_lam*cos_phi
    
    xyz[1] = (a * sin_lam) / tmp_den + lla[2] * sin_lam*cos_phi
    
    tmp2 = np.sqrt(1 - e * e * sin_phi * sin_phi)
    xyz[2] = (a * tmp * sin_phi) / tmp2 + lla[2] * sin_phi
    
    return xyz


# Same as lla2xyz but now in Astropy
# Not used because it is too slow
def lla2xyz_astropy(lla):
    xyz = EarthLocation.from_geodetic(degrees(lla[1]), degrees(lla[0]), lla[2], 'WGS84')
    return list(xyz.value)


# Convert from ECF cartesian coordinates to latitude, longitude and height, taking
# into account the Earth flattening
#
# Ref. Converts from WGS-84 X, Y and Z rectangular co-ordinates to latitude,
#      longitude and height. For more information: Simo H. Laurila,
#      "Electronic Surveying and Navigation", John Wiley & Sons (1976).
#
# @param XYZ ECEF XYZ coordinates in m
# @param LLA Latitude rad/[-PI/2,PI/2] positive N, Longitude rad/[-PI,PI] positive E, height above ellipsoid
@jit(nopython=True)
def xyz2lla(xyz):

    x2 = xyz[0] * xyz[0]
    y2 = xyz[1] * xyz[1]
    z2 = xyz[2] * xyz[2]
    
    sma = 6378137.0000  # earth radius in meters
    smb = 6356752.3142  # earth semi minor in meters
    
    e = np.sqrt(1.0 - (smb / sma)*(smb / sma))
    b2 = smb*smb
    e2 = e*e
    ep = e * (sma / smb)
    r = np.sqrt(x2 + y2)
    r2 = r*r
    E2 = sma * sma - smb*smb
    big_f = 54.0 * b2*z2
    G = r2 + (1.0 - e * e) * z2 - e * e*E2
    small_c = (e * e * e * e * big_f * r2) / (G * G * G)
    s = np.power(1.0 + small_c + np.sqrt(small_c * small_c + 2.0 * small_c), 1.0 / 3.0)
    P = big_f / (3.0 * (s + 1.0 / s + 1.0)*(s + 1.0 / s + 1.0) * G * G)
    Q = np.sqrt(1 + 2 * e2 * e2 * P)
    ro = -(P * e * e * r) / (1.0 + Q) + np.sqrt((sma * sma / 2.0)*(1.0 + 1.0 / Q) -
                                             (P * (1 - e * e) * z2) / (Q * (1.0 + Q)) - P * r2 / 2.0)
    tmp = (r - e * e * ro)*(r - e * e * ro)
    U = np.sqrt(tmp + z2)
    V = np.sqrt(tmp + (1 - e * e) * z2)
    zo = (b2 * xyz[2]) / (sma * V)
    
    height = U * (1 - b2 / (sma * V))
    
    lat = np.arctan((xyz[2] + ep * ep * zo) / r)
    
    temp = np.arctan(xyz[1] / xyz[0])
    
    if xyz[0] >= 0:
        lon = temp
    elif xyz[0] < 0 and xyz[1] >= 0:
        lon = PI + temp
    else:
        lon = temp - PI
    
    return [lat, lon, height]


# Compute satellite position given Kepler set in ECI reference frame
#
# @param MJD_Requested Modified Julian date for wanted position
# @param Kepler Class object with kepler elements in radians and meters
# double Epoch_MJD
# double Semi_Major_Axis
# double Eccentricity
# double Inclination
# double RAAN
# double Arg_Of_Perigee
# double Mean_Anomaly
# @param Out POS_VEL_XYZ in m and m/s
@jit(nopython=True)
def kep2xyz(mjd_requested,
            kepler_epoch_mjd, kepler_semi_major_axis, kepler_eccentricity, kepler_inclination,
            kepler_right_ascension, kepler_arg_perigee, kepler_mean_anomaly):

    pos = np.zeros(3)
    vel = np.zeros(3)

    # Compute radius, corrected right ascension and mean anomaly
    time_from_ref = (mjd_requested - kepler_epoch_mjd) * 86400
    mean_motion = np.sqrt(GM_EARTH / (np.power(kepler_semi_major_axis, 3)))  #radians/s
    mean_anomaly = kepler_mean_anomaly + (mean_motion * time_from_ref)

    # if eccentricity equals zero eccentric anomaly is equal mean anomaly
    if kepler_eccentricity == 0:
        eccentric_anomaly = mean_anomaly
    else:  # Newton Rhapson method
        k = 0
        big_e = np.zeros(50)

        big_e[1] = mean_anomaly
        big_e[0] = mean_anomaly * 2

        while fabs(big_e[k + 1] / big_e[k] - 1) > 1e-15:  # Newton Rhapson
            k += 1
            big_e[k + 1] = big_e[k] - (
                        (big_e[k] - kepler_eccentricity * np.sin(big_e[k]) - mean_anomaly) /
                        (1 - kepler_eccentricity * np.cos(big_e[k])))
        eccentric_anomaly = big_e[k + 1]

    radius = kepler_semi_major_axis * (1 - kepler_eccentricity * np.cos(eccentric_anomaly))

    sin_nu_k = ((np.sqrt(1 - kepler_eccentricity * kepler_eccentricity)) * np.sin(eccentric_anomaly)) / \
               (1 - kepler_eccentricity * np.cos(eccentric_anomaly))
    cos_nu_k = (np.cos(eccentric_anomaly) - kepler_eccentricity) / (1 - kepler_eccentricity * np.cos(eccentric_anomaly))
    true_anomaly = np.arctan2(sin_nu_k, cos_nu_k)
    arg_of_lat = kepler_arg_perigee + true_anomaly

    # Apply rotation from elements to xyz
    xyk1 = radius * np.cos(arg_of_lat)
    xyk2 = radius * np.sin(arg_of_lat)
    pos[0] = np.cos(kepler_right_ascension) * xyk1 + -np.cos(kepler_inclination) * np.sin(kepler_right_ascension) * xyk2
    pos[1] = np.sin(kepler_right_ascension) * xyk1 + np.cos(kepler_inclination) * np.cos(kepler_right_ascension) * xyk2
    pos[2] = np.sin(kepler_inclination) * xyk2

    # Compute velocity components
    factor = np.sqrt(GM_EARTH / kepler_semi_major_axis / (1 - np.power(kepler_eccentricity, 2)))
    sin_w = np.sin(kepler_arg_perigee)
    cos_w = np.cos(kepler_arg_perigee)
    sin_o = np.sin(kepler_right_ascension)
    cos_o = np.cos(kepler_right_ascension)
    sin_i = np.sin(kepler_inclination)
    cos_i = np.cos(kepler_inclination)
    sin_t = np.sin(true_anomaly)
    cos_t = np.cos(true_anomaly)

    l1 = cos_w * cos_o - sin_w * sin_o*cos_i
    m1 = cos_w * sin_o + sin_w * cos_o*cos_i
    n1 = sin_w*sin_i
    l2 = -sin_w * cos_o - cos_w * sin_o*cos_i
    m2 = -sin_w * sin_o + cos_w * cos_o*cos_i
    n2 = cos_w*sin_i

    vel[0] = factor * (-l1 * sin_t + l2 * (kepler_eccentricity + cos_t))
    vel[1] = factor * (-m1 * sin_t + m2 * (kepler_eccentricity + cos_t))
    vel[2] = factor * (-n1 * sin_t + n2 * (kepler_eccentricity + cos_t))

    return pos, vel


# Iterates to solution of eccentric_anomaly using numerical solution
# Only necessary if eccentricity is unequal to 0
# 
# Ref. http:en.wikipedia.org/wiki/Newton's_method
# 
# @param MeanAnomaly mean anomaly (radians)
# @param Eccentricity eccentricity (-)
# @return  Eccentric anomaly (radians)
def newton_raphson (mean_anomaly, eccentricity):
    k = 0
    big_e = np.zeros(50)

    big_e[1] = mean_anomaly
    big_e[0] = mean_anomaly * 2

    while fabs(big_e[k + 1] / big_e[k] - 1) > 1e-15:
        k += 1
        big_e[k + 1] = big_e[k]-((big_e[k] - eccentricity * sin(big_e[k]) - mean_anomaly) / (1 - eccentricity * cos(big_e[k])))

    return big_e[k + 1]


# Rotate vector on z-axis
#
# @param Angle Rotation angle (radians/double)
# @param Vector Position input vector (1:6) (meters/double)
# @param Out Rotated position vector (1:6) (meters/double)
@jit(nopython=True)
def spin_vector(angle, pos_vec, vel_vec):

    pos = np.zeros(3)
    vel = np.zeros(3)

    # Compute angles to save time
    cosst = np.cos(angle)
    sinst = np.sin(angle)
    
    pos[0] = cosst * pos_vec[0] - sinst * pos_vec[1]
    pos[1] = sinst * pos_vec[0] + cosst * pos_vec[1]
    pos[2] = pos_vec[2]
    
    vel[0] = cosst * vel_vec[0] - sinst * vel_vec[1]
    vel[1] = sinst * vel_vec[0] + cosst * vel_vec[1]
    vel[2] = vel_vec[2]
    
    return pos, vel


# Compute Azimuth and Elevation from User to Satellite assuming the Earth is a perfect sphere
#
# @param Xs Satellite position in ECEF (m/double)
# @param Xu User position in ECEF (m/double)
# @param AzEl Azimuth [0] and Elevation [1] (radians/double)
@jit(nopython=True)
def calc_az_el(xs, xu):

    az_el = np.zeros(2)
    e3by3 = np.zeros((3,3))
    d = np.zeros(3)

    x = xu[0]
    y = xu[1]
    z = xu[2]
    p = sqrt(x * x + y * y)

    R = sqrt(x * x + y * y + z * z)
    e3by3[0][0] = -(y / p)
    e3by3[0][1] = x / p
    e3by3[0][2] = 0.0
    e3by3[1][0] = -(x * z / (p * R))
    e3by3[1][1] = -(y * z / (p * R))
    e3by3[1][2] = p / R
    e3by3[2][0] = x / R
    e3by3[2][1] = y / R
    e3by3[2][2] = z / R

    # matrix multiply vector from user */
    for k in range(3):
        d[k] = 0.0
        for i in range(3):
            d[k] += (xs[i] - xu[i]) * e3by3[k][i]

    s = d[2] / sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if s == 1.0:
        az_el[1] = 0.5 * PI
    else:
        az_el[1] = atan(s / sqrt(1.0 - s * s))

    if d[1] == 0.0 and d[0] > 0.0:
        az_el[0] = 0.5 * PI
    elif d[1] == 0.0 and d[0] < 0.0:
        az_el[0] = 1.5 * PI
    else:
        az_el[0] = atan(d[0] / d[1])
        if d[1] < 0.0:
            az_el[0] += PI
        elif d[1] > 0.0 and d[0] < 0.0:
            az_el[0] += 2.0 * PI

    return az_el[0], az_el[1]


# Computes two intersection points for a line and a sphere
# Returns false if the line does not intersect the sphere
# Returns two equal intersection points if the line is tangent to the sphere
# Returns two different intersection points if the line gos through the sphere
#
# @param X1 Point 1 on the line
# @param X2 Point 2 on the line
# @param SphereRadius Radius of the sphere
# @param SphereCenterX Center point of the sphere
# @param iX1 Intersection point one
# @param iX2 Intersection point two
# @return Returns false if the line does not intersect the sphere
@jit(nopython=True)
def line_sphere_intersect(x1, x2, sphere_radius, sphere_center):

    intersect = True

    i_x1 = np.zeros(3)
    i_x2 = np.zeros(3)

    a = (x2[0] - x1[0]) * (x2[0] - x1[0]) + (x2[1] - x1[1]) * (x2[1] - x1[1]) + (x2[2] - x1[2]) * (x2[2] - x1[2])

    b = 2.0 * ((x2[0] - x1[0]) * (x1[0] - sphere_center[0]) + (x2[1] - x1[1]) * (x1[1] - sphere_center[1]) +
               (x2[2] - x1[2]) * (x1[2] - sphere_center[2]))

    c = sphere_center[0] * sphere_center[0] + sphere_center[1] * sphere_center[1] + \
        sphere_center[2] * sphere_center[2] + x1[0] * x1[0] + x1[1] * x1[1] + x1[2] * x1[2] - \
        2.0 * (sphere_center[0] * x1[0] + sphere_center[1] * x1[1] + sphere_center[2] * x1[2]) - \
        sphere_radius * sphere_radius

    in_sqrt = (b * b - 4.0 * a * c)

    # No intersection
    if in_sqrt < 0:
        intersect = False
    else:
        # Tangent (0) or intersection (+)
        u1 = (-b + sqrt(in_sqrt)) / 2.0 / a
        u2 = (-b - sqrt(in_sqrt)) / 2.0 / a

        # Intersection points
        i_x1[0] = x1[0] + u1 * (x2[0] - x1[0])
        i_x1[1] = x1[1] + u1 * (x2[1] - x1[1])
        i_x1[2] = x1[2] + u1 * (x2[2] - x1[2])

        i_x2[0] = x1[0] + u2 * (x2[0] - x1[0])
        i_x2[1] = x1[1] + u2 * (x2[1] - x1[1])
        i_x2[2] = x1[2] + u2 * (x2[2] - x1[2])

    return intersect, i_x1, i_x2


# Compute satellite visibility contour for a certain ElevationMask of the users,
# assuming the Earth is a perfect sphere
#
# REFERENCE
# Space Mission Analysis and Design, 3rd edition (Space Technology Library)
# W. Larson and J. Wertz
#
# @param LLA Sub Satellite Point Latitude, Longitude, Altitude rad/m
# @param ElevationMask User to satellite elevation mask (rad)
# @param Contour Array with Lat Lon points on the satellite visibility contour lat/lon in (rad)
def sat_contour(lla, elevation_mask):

    step_size = 0.3

    contour = np.zeros((math.ceil(360.0/step_size), 2))

    lat_s = lla[0]
    lon_s = lla[1]

    lam=0
    if elevation_mask == 0:
        lam = acos(R_EARTH / (R_EARTH + lla[2]))
    else:
        lam = PI / 2 - elevation_mask - asin(cos(elevation_mask) * R_EARTH / (R_EARTH + lla[2]))


    # Problem detected at Az=180 for acos, so step size set to 0.3 iso 0.5

    lat_t, lon_t = 0, 0
    delta_lon = 0
    cnt = 0
    for idx_az in np.arange(0.0, 360.0, step_size):

        try:
            az = radians(idx_az)
            lat_t_acc = acos(cos(lam) * sin(lat_s) + sin(lam) * cos(lat_s) * cos(az))
            lat_t = PI / 2 - lat_t_acc
            delta_lon = acos((cos(lam)-(sin(lat_s) * sin(lat_t))) / (cos(lat_s) * cos(lat_t)))
        except:
            pass
        if idx_az < 180:
            lon_t = lon_s + delta_lon
        else:
            lon_t = lon_s - delta_lon

        if lon_t > PI:
            lon_t = lon_t - 2 * PI
        if lon_t < -PI:
            lon_t = lon_t + 2 * PI

        contour[cnt, :] = [lat_t, lon_t]

        cnt += 1

    return contour

@jit(nopython=True)  # paralellism did not work, only for loops...
def dist_point_plane(point_q, plane_normal):  # not really distance but if point is more than 90 deg away from normal
    return np.dot(point_q, plane_normal)

@jit(nopython=True)  # paralellism did not work, only for loops...
def test_point_within_pyramid(point_q, planes_h):
    for i in range(4):
        if dist_point_plane(point_q, planes_h[i,:]) > 0.0:  # on bad side of plane, faster implementation
            return False
    return True


def make_unit(x):
    """Normalize entire input to norm 1. Not what you want for 2D arrays!"""
    return x / norm(x)


def x_parallel_v(x, v):
    """Project x onto v. Result will be parallel to v."""
    # (x' * v / norm(v)) * v / norm(v)
    # = (x' * v) * v / norm(v)^2
    # = (x' * v) * v / (v' * v)
    return dot(x, v) / dot(v, v) * v


def x_perpendicular_v(x, v):
    """Component of x orthogonal to v. Result is perpendicular to v."""
    return x - x_parallel_v(x, v)


def x_project_on_v(x, v):
    """Project x onto v, returning parallel and perpendicular components
    >> d = xProject(x, v)
    >> np.allclose(d['par'] + d['perp'], x)
    True
    """
    par = x_parallel_v(x, v)
    perp = x - par
    return {'par': par, 'perp': perp}


def rot_vec_vec(a, b, theta):
    """Rotate vector a about vector b by theta radians."""
    # Thanks user MNKY at http: #math.stackexchange.com/a/1432182/81266
    proj = x_project_on_v(a, b)
    w = cross(b, proj['perp'])
    return proj['par'] + proj['perp'] * cos(theta) + norm(proj['perp']) * make_unit(w) * sin(theta)


@jit(nopython=True)
def det_swath_radius(alt, inc_angle, r_earth): # altitude in [m], incidence angle alfa in [radians]
    oza = det_oza_fast(alt, r_earth, inc_angle)
    return (oza - inc_angle)*r_earth # in [m]


@jit(nopython=True)
def det_oza_fast(alt, r_earth, inc_angle):  # altitude and R_EARTH in [m], incidence angle  in [radians]
    oza = asin((r_earth + alt) / r_earth * sin(inc_angle))
    return oza


@jit(nopython=True)
def earth_angle_beta(swath_radius, r_earth):
    # Returns the Earth beta angle in [radians], from swath radius in [m]
    beta = asin(swath_radius / r_earth)
    return beta


@jit(nopython=True)  # paralellism did not work, only for loops...
def angle_two_vectors(u, v, norm_u, norm_v):
    # Returns angle in [radians]
    # Pre computed the norm of the second vector
    # c = np.dot(u, v) / norm_u / norm_v
    c = (u[0]*v[0]+u[1]*v[1]+u[2]*v[2])/norm_u/norm_v
    if c>1:
        c=1
    if c<-1:
        c=-1
    #result = np.arccos(np.clip(c, -1, 1))
    result = np.arccos(c)
    return result


# Compute numerically the solution to the incidence angle alfa from a swath width from nadir
# swath width in [m]
# radius Earth in [m]
# altitude satellite in [m]
# return incidence angle in [rad]
@jit(nopython=True)
def incl_from_swath(swath_w, r_earth, alt):
    solution = atan(sin(swath_w/r_earth) / ((r_earth + alt)/r_earth - cos(swath_w/r_earth)))
    return solution


# Compute Earth radius at latitude
# input lat [rad]
# output radius R [m]
def earth_radius_lat(lat):
    r1 = 6378137  # radius at equator in [m]
    r2 = 6356752  # radius at pole in [m]
    radius = sqrt((r1*r1*r1*r1*pow(cos(lat),2) + r2*r2*r2*r2*pow(sin(lat),2)) /
                  (r1*r1*pow(cos(lat),2) + r2*r2*pow(sin(lat),2)))
    return radius


@jit(nopython=True)
def check_users_in_plane(user_metric, user_pos, planes, cnt_epoch):
    n_users = len(user_metric)
    for idx_user in range(n_users):
        if test_point_within_pyramid(user_pos[idx_user, :], planes):
            user_metric[idx_user,cnt_epoch] = 1  # Within swath
    return user_metric[:,cnt_epoch]


@jit(nopython=True)
def check_users_from_nadir(user_metric, user_pos, sat_pos, earth_angle_swath, cnt_epoch):
    norm_sat = norm(sat_pos)
    n_users = len(user_metric)
    for idx_user in range(n_users):
        norm_user = norm(user_pos[idx_user,:])
        angle_user_zenith = angle_two_vectors(user_pos[idx_user, 0:3], sat_pos, user_pos[idx_user,3], norm_sat)
        if angle_user_zenith < earth_angle_swath:
            user_metric[idx_user,cnt_epoch] = 1  # Within swath
    return user_metric[:,cnt_epoch]


# Convert string True/False to boolean
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# Compute gas attenuation
# Gas attenuation models are given in ITU-R P676-11.
# The model provided is computationally intensive and requires a lot of tabulated data.
# In order to implement this with a low computational burden in a short time, a simplified-statistical model has been derived.
# The elevation dependences have been mapped for each frequency with an exponential fit.
# The result is a coefficients table:
# Input: frequency [Hz]
# Input: elevation [rad]
# Output: attenuation [dB]
def comp_gas_attenuation(frequency, elevation):  # TODO use ITUR py package...

    frequency = frequency / 1e9  # assume freq GHz, elevation deg
    elevation = degrees(elevation)  # assume freq GHz, elevation deg

    x = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]
    y = [1.145, 1.4216, 1.5098, 1.5701, 1.6304, 1.779, 2.1214, 3.4171, 10.679, 13.972, 9.9376, 11.858, 16.777, 65.193,
         6527, 71.106]
    tck = interpolate.splrep(x, y, s=0)
    a = interpolate.splev(frequency, tck, der=0)  # compute attenuation

    if elevation > 0.1:  # model unreliable less than 0.1 deg
        gas_attenuation = a * np.power(elevation, -0.8622)
    else:
        gas_attenuation = 10  # safe clause upper limit (ITUR P372-10 plots)

    return gas_attenuation


# Taken from Gérard Maral, Michel Bousquet. Satellite Communications Systems. John Wiley & Sons. 4th Edition, 2002.
# Input:    Frequency   Hz
# Input:    Elevation   rad
# Input:    p_exceed    %
# Input:    Latitude    rad
# Input:    Height      m
# Input:    Rainfall Rate   mm/hr
# Output:   Rain attenuation    dB
def comp_rain_attenuation(frequency, elevation, latitude, height, rain_p_exceed, rainfall_rate, rain_height):

    frequency = frequency/1e9  # Convert to GHz
    height = height / 1000 # Convert to m
    latitude = degrees(latitude) # Convert to deg

    kh_ref = [0.0000259,0.0000847,0.0001071,0.0007056,0.001915,0.004115,0.01217,0.02386,0.04481,0.09164,0.1571,
              0.2403,0.3374,0.4431]
    alphah_ref = [0.9691,1.06664,1.6009,1.59,1.481,1.3905,1.2571,1.1825,1.1233,1.0568,0.9991,0.9485,0.9047,0.8673]
    kv_ref = [0.0000308,0.0000998,0.0002461,0.0004878,0.001425,0.00345,0.01129,0.02455,0.05008,0.09611,0.1533,
              0.2291,0.3224,0.4274]
    alphav_ref = [0.8592,0.949,1.2476,1.5728,1.4745,1.3797,1.2156,1.1216,1.044,0.9847,0.9491,0.9129,0.8761,0.8421]

    freq_values = [1,2,4,6,7,8,10,12,15,20,25,30,35,40]

    # calculate RAIN HEIGHT by latitude [Charlesworth model/ ITUR P.839]
    if rain_height == None:
        hrain=0  #some regions have no rain, these  are not be affected by the clauses/
        if 23 < latitude < 89:
            hrain=5-0.075*(latitude-23)
        elif 0 < latitude <= 23:
            hrain=3.2-0.075*(latitude-35)
        elif -21 < latitude <= 0:
            hrain=5
        elif -71 < latitude <= -21:
            hrain=5+0.1*(latitude+21)
    else:  # Get it from the configuration file
        hrain = rain_height

    Ls=(hrain-(height))/(sin(elevation))   #in km
    Lg=Ls*cos(elevation)  #calculate SLANT distance and horizontal projection

    tck = interpolate.splrep(freq_values, kh_ref, s=0)  # interpolate k horizontal
    kh = interpolate.splev(frequency, tck, der=0)

    tck = interpolate.splrep(freq_values, alphah_ref, s=0)  # interpolate alpha horizontal
    alphah = interpolate.splev(frequency, tck, der=0)

    tck = interpolate.splrep(freq_values, kv_ref, s=0)  # interpolate k vertical
    kv = interpolate.splev(frequency, tck, der=0)

    tck = interpolate.splrep(freq_values, alphav_ref, s=0)  # interpolate alpha vertical
    alphav = interpolate.splev(frequency, tck, der=0)

    # Calculate Specific attenuation, horizontal, vertical and circular (reference)
    specific_att_h = kh * np.power(rainfall_rate,alphah)
    specific_att_v = kv * np.power(rainfall_rate,alphav)
    specific_att = 0.5 * (specific_att_v+specific_att_h)  # circular polarization has to be averaged

    # calculate reduction factor r
    aux1 = Lg * specific_att/frequency
    aux2 = np.power(aux1,0.5)
    aux3 = np.exp(-2*Lg)
    r = 1/(1+0.78*aux2-0.38*(1-aux3))

    # calculate zheta
    zeta = atan(hrain-(height)/(Lg*r))

    # select LR
    if zeta>elevation:
        Lr = Lg * r / cos(elevation)
    else:
        Lr=(hrain-(height))/(sin(elevation))

    # select Chi
    if -36 < latitude < 36:
        Chi = 36 - latitude
    else:
        Chi = 0

    # calculate vertical adjustment factor
    aux1=1-np.exp(-(elevation*180/(PI*(1+Chi))))   #the expression needed degs
    aux2=Lr*specific_att
    aux3=np.power(aux2,0.5)/(frequency*frequency)
    v=1/(1+np.power(sin(elevation),0.5)*(31*aux1*aux3-0.45))

    # compute effective path length
    Le=Lr*v

    # compute attenuation for the 0.01 exceed
    att_001=Le*specific_att

    # compute beta
    if np.abs(latitude)>=36 or rain_p_exceed>=1:
        beta = 0
    elif rain_p_exceed<1 and abs(latitude)<36 and elevation>0.436:
        beta = (abs(latitude)-36)*(-0.005)
    else:
        beta = (abs(latitude)-36)*(-0.005)+1.8-4.25*sin(elevation)

    # compute attenuation correspondent to p
    aux1 = 0.655+0.033*np.log(rain_p_exceed)-0.045*np.log(att_001)-beta*(1-rain_p_exceed)*sin(elevation)
    aux2 = np.power(rain_p_exceed/0.01,-aux1)
    rain_attenuation = att_001*aux2

    return rain_attenuation


# Input frequency Hz
# Input elevation rad
# Output brightness temp sky K
def temp_brightness(frequency, elevation):

    frequency = frequency/1e9  # Convert to GHz
    elevation = degrees(elevation)  # Convert to degrees

    freq_values_ref = [2,5,10,15,20,25,30,35,40,45,50,55,60]
    c_parameter_ref = [72.178,79.19,108.81,198.17,464.43,483.02,394.63,444.76,589.25,619.48,644.86,290,290]
    d_parameter_ref = [-0.8434,-0.8239,-0.814,-0.814,-0.7704,-0.7573,-0.7599,-0.7101,-0.7501,-0.633,-0.4576,0,0]

    tck = interpolate.splrep(freq_values_ref, c_parameter_ref, s=0)  # for c value interpolation
    c = interpolate.splev(frequency, tck, der=0)

    tck = interpolate.splrep(freq_values_ref, d_parameter_ref, s=0)  # for d value interpolation
    d = interpolate.splev(frequency, tck, der=0)

    if elevation>3:  #arbitrary limit, model unreliable less than 5 deg and singular at 0.
        temperature = c * np.power(elevation,d)
    else:
        temperature = 290  # safe upper limit (ITUR P372-10 plots)

    return temperature


# Input Modulation type BPSK/QPSK
# Input BER in multiples of 10
# Input datarate in bps
# Output CN0 required in dB
def comp_cn0_required(modulation, ber, datarate):

    ber = np.log10(ber)
    # Valid for BPSK and QPSK
    if modulation == 'BPSK' or modulation == 'QPSK':
        dict = {-3:6.8,
                -4:8.4,
                -5:9.6,
                -6:10.5,
                -7:11.3,
                -8:12.0,
                -9:12.6}
        cn0_req = dict[ber]+10*np.log10(datarate)
        return cn0_req
    else:
        return 100



