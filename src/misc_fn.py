import math
from math import sin, cos, atan2, atan, fabs, acos
import time
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy import time
from astropy import coordinates, units as u
from math import tan, sqrt, asin, degrees, radians
from numpy import dot, arccos, cross, sin, cos
from numpy.linalg import norm
from numba import jit, float64 # todo test new modes in numba: njit, fastmath, cache, paralel...
from scipy import interpolate
from scipy.special import jv  # Bessel function of 1st kind

# Modules from project
from constants import R_EARTH, PI, GM_EARTH
import logging_svs as ls

from multiprocessing import Process

def benchmark(func):
    """
    A decorator that prints the time a function takes to execute.
    """
    def wrapper(*args, **kwargs):
        t = time.process_time()
        res = func(*args, **kwargs)
        t_sec = round((time.process_time()-t) % 60,1)
        t_min = int((time.process_time()-t)/ 60)
        ls.logger.info(f'Application function {func.__name__} execution time {t_min} [min] {t_sec} [sec]')
        return res
    return wrapper


def plane_normal(a,b):
    """
    :param a: 3d vector to point on plane
    :param b: 3d vector to point on plane
    :return:  3d vector normal of the plane
    :rtype: list
    Returns the plane normal for a plane which has the origin in its plane
    """
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c


def yyyy_doy2mjd(yyyy, doy):
    """
    :param yyyy: 4 digit year
    :param doy: decimal day of year
    :return: Modified Julian Date
    :rtype: float
    Converts year, doy to MJD, taken from from SP3 library B.Remondi.
    """

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


def mjd2gmst(mjd_requested):
    """
    :param mjd_requested: MJD to be converted
    :return: Greenwich Mean Sidereal Time (radians) [0 to 2PI]
    :rtype: float
    Compute Greenwich Mean Sidereal Time from Modified Julian Date.
    Ref. Time functions from SP3 library B.Remondi.
    """

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


@jit(nopython=True)
def lla2xyz(lla):
    """
    :param lla: LLA Latitude rad/[-PI/2,PI/2] positive N, Longitude rad/[-PI,PI] positive E, height above ellipsoid
    :return: XYZ ECEF XYZ coordinates in m
    Convert from latitude, longitude and height to ECF cartesian coordinates, taking into account the Earth flattening
    Ref. Understanding GPS: Principles and Applications,E. Kaplan, Editor, Artech House Publishers, Boston 1996.
    """

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


def lla2xyz_astropy(lla):
    """
    :param lla: LLA Latitude rad/[-PI/2,PI/2] positive N, Longitude rad/[-PI,PI] positive E, height above ellipsoid
    :return: XYZ ECEF XYZ coordinates in m
    Same as lla2xyz but now in Astropy, not used because it is too slow
    """
    xyz = EarthLocation.from_geodetic(degrees(lla[1]), degrees(lla[0]), lla[2], 'WGS84')
    return list(xyz.value)


#
#
# Ref. Converts from WGS-84 X, Y and Z rectangular co-ordinates to latitude,
#      longitude and height. For more information: Simo H. Laurila,
#      "Electronic Surveying and Navigation", John Wiley & Sons (1976).

@jit(nopython=True)
def xyz2lla(xyz):
    """
    :param xyz: ECEF XYZ coordinates in m
    :return: LLA Latitude rad/[-PI/2,PI/2] positive N, Longitude rad/[-PI,PI] positive E, height above ellipsoid
    Convert from ECF cartesian coordinates to latitude, longitude and height, taking into account the Earth flattening.
    Ref. Converts from WGS-84 X, Y and Z rectangular co-ordinates to latitude,longitude and height.
    For more information: Simo H. Laurila, "Electronic Surveying and Navigation", John Wiley & Sons (1976).
    """

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


@jit(nopython=True)
def kep2xyz(mjd_requested,
            kepler_epoch_mjd, kepler_semi_major_axis, kepler_eccentricity, kepler_inclination,
            kepler_right_ascension, kepler_arg_perigee, kepler_mean_anomaly):
    """
    :param mjd_requested:
    :param kepler_epoch_mjd:
    :param kepler_semi_major_axis:
    :param kepler_eccentricity:
    :param kepler_inclination:
    :param kepler_right_ascension:
    :param kepler_arg_perigee:
    :param kepler_mean_anomaly:
    :return: pos,vel 3d vectors in m and m/s
    Compute satellite position given Kepler set in ECI reference frame
    """

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


def newton_raphson (mean_anomaly, eccentricity):
    """
    :param mean_anomaly: MeanAnomaly mean anomaly (radians)
    :param eccentricity: Eccentricity eccentricity (-)
    :return: Eccentric anomaly (radians)
    """
    k = 0
    big_e = np.zeros(50)

    big_e[1] = mean_anomaly
    big_e[0] = mean_anomaly * 2

    while fabs(big_e[k + 1] / big_e[k] - 1) > 1e-15:
        k += 1
        big_e[k + 1] = big_e[k]-((big_e[k] - eccentricity * sin(big_e[k]) - mean_anomaly) / (1 - eccentricity * cos(big_e[k])))

    return big_e[k + 1]


@jit(nopython=True)
def spin_vector(angle, pos_vec, vel_vec):
    """
    :param angle: Angle Rotation angle (radians/double)
    :param pos_vec: Vector Position input vector (1:3) (meters/double)
    :param vel_vec: Vector Velocity input vector (1:3) (meters/double)
    :return: Rotated position and velocity vector (1:3) (meters/double)
    Rotate position and velocity around z-axis
    """

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


def calc_az_el(xs, xu):
    """
    :param xs: Xs Satellite position in ECEF (m/double)
    :param xu: Xu User position in ECEF (m/double)
    :return: Az, El Azimuth and Elevation (radians/double)
    Compute Azimuth and Elevation from User to Satellite assuming the Earth is a perfect sphere
    """
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


def line_sphere_intersect(x1, x2, sphere_radius, sphere_center):
    """
    :param x1: Point 1 on the line
    :param x2: Point 2 on the line
    :param sphere_radius: SphereRadius Radius of the sphere
    :param sphere_center: Center point of the sphere
    :return: Returns false if the line does not intersect the sphere, Intersection point one and two
    Computes two intersection points for a line and a sphere
    Returns false if the line does not intersect the sphere
    Returns two equal intersection points if the line is tangent to the sphere
    Returns two different intersection points if the line gos through the sphere
    """

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


def sat_contour(lla, elevation_mask):
    """
    :param lla: Sub Satellite Point Latitude, Longitude, Altitude rad/m
    :param elevation_mask: ElevationMask User to satellite elevation mask (rad)
    :return: Contour np.array with Lat Lon points on the satellite visibility contour lat/lon in (rad)
    Compute satellite visibility contour for a certain ElevationMask of the users,assuming the Earth is a perfect sphere
    Ref.Space Mission Analysis and Design, 3rd edition (Space Technology Library), W. Larson and J. Wertz
    """

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


@jit(nopython=True)
def dist_point_plane(point_q, plane_normal):
    """
    :param point_q: point q 3d vector
    :param plane_normal: plane normal 3d vector
    :return: dot product between point q and normal, measure of angle and length both vectors
    """
    return np.dot(point_q, plane_normal)


@jit(nopython=True)
def test_point_within_pyramid(point_q, planes_h):
    """
    :param point_q: point q
    :param planes_h: 4 plane normals defining a pyramid
    :return: True if point is within pyramid
    Compute if point is within pyramid with top at origin
    """
    for i in range(4):
        if dist_point_plane(point_q, planes_h[i,:]) > 0.0:  # on bad side of plane, faster implementation
            return False
    return True


def make_unit(x):
    """
    :param x: vector
    :return: normalized vector to length 1
    Normalize a vector
    """
    return x / norm(x)


def x_parallel_v(x, v):
    """
    :param x: vector x
    :param v: vector v
    :return: projection of x onto v, result will be parallel to v
    """
    # (x' * v / norm(v)) * v / norm(v)
    # = (x' * v) * v / norm(v)^2
    # = (x' * v) * v / (v' * v)
    return dot(x, v) / dot(v, v) * v


def x_perpendicular_v(x, v):
    """
    :param x: vector x
    :param v: vector v
    :return: projection of x orthogonal to v, result will be perpendicular to v
    """
    return x - x_parallel_v(x, v)


def x_project_on_v(x, v):
    """
    :param x: vector x
    :param v: vector v
    :return: dict with paralel and perpendicular components
    """
    par = x_parallel_v(x, v)
    perp = x - par
    return {'par': par, 'perp': perp}


def rot_vec_vec(a, b, theta):
    """
    :param a: 3d vector a
    :param b: 3d vector b
    :param theta: rotation angle in radians
    :return: rotated vector a
    Rotate vector a about vector b by theta radians.
    """
    proj = x_project_on_v(a, b)
    w = cross(b, proj['perp'])
    return proj['par'] + proj['perp'] * cos(theta) + norm(proj['perp']) * make_unit(w) * sin(theta)


@jit(nopython=True)
def det_swath_radius(alt, inc_angle, r_earth):
    """
    :param alt: altitude in [m]
    :param inc_angle: incidence angle alfa in [radians]
    :param r_earth: radius Earth in [m]
    :return: compute swath radius for conical scanner in [m]
    """
    # ,
    oza = det_oza_fast(alt, r_earth, inc_angle)
    return (oza - inc_angle)*r_earth


@jit(nopython=True)
def det_oza_fast(alt, r_earth, inc_angle):
    """
    :param alt: altitude in [m]
    :param r_earth: radius Earth in [m]
    :param inc_angle: incidence angle in [rad]
    :return: observation zenith angle in [rad]
    compute OZA from altitude, radius_earth and incidence angle
    """
    oza = asin((r_earth + alt) / r_earth * sin(inc_angle))
    return oza


@jit(nopython=True)
def earth_angle_beta(swath_radius, r_earth):
    """
    :param swath_radius: swath radius in [m]
    :param r_earth: radius Earth in [m]
    :return: Earth beta angle in [radians]
    """
    beta = asin(swath_radius / r_earth)
    return beta


@jit(nopython=True)
def angle_two_vectors(u, v, norm_u, norm_v):
    """
    :param u: 3d vector
    :param v: 3d vector
    :param norm_u: norm of u
    :param norm_v: norm of v
    :return: angle between u and v in [rad]
    Assumes norms are pre-computed
    """
    c = (u[0]*v[0]+u[1]*v[1]+u[2]*v[2])/norm_u/norm_v
    if c>1:
        c=1
    if c<-1:
        c=-1
    #result = np.arccos(np.clip(c, -1, 1))
    result = np.arccos(c)
    return result


@jit(nopython=True)
def incl_from_swath(swath_w, r_earth, alt):
    """
    :param swath_w: swath width in [m]
    :param r_earth: radius Earth in [m]
    :param alt: altitude satellite in [m]
    :return: incidence angle in [rad]
    Compute numerically the solution to the incidence angle alfa from a swath width from nadir
    """
    solution = atan(sin(swath_w/r_earth) / ((r_earth + alt)/r_earth - cos(swath_w/r_earth)))
    return solution


def earth_radius_lat(lat):
    """
    :param lat: [rad]
    :return: radius R [m]
    Compute Earth radius at latitude from elliptical shape
    """
    r1 = 6378137  # radius at equator in [m]
    r2 = 6356752  # radius at pole in [m]
    radius = sqrt((r1*r1*r1*r1*pow(cos(lat),2) + r2*r2*r2*r2*pow(sin(lat),2)) /
                  (r1*r1*pow(cos(lat),2) + r2*r2*pow(sin(lat),2)))
    return radius


#
# def check_users_in_plane(user_pos, planes, shared_array):
#     """
#     :param user_pos: user position ECEF
#     :param planes: normals of the planes
#     :param shared_array: shared array
#     :return: user-metric
#     Does not work well since processes take too much overhead to run, therefore normal numba version preferred
#     """
#
#     num_processors = 8  # for mac pro
#     process_list = []
#     for i in range(num_processors):
#         process_list.append(Process(target=process_users, args=(shared_array, num_processors, i, user_pos, planes)))
#         process_list[i].start()
#
#     for i in range(num_processors):
#         process_list[i].join()
#
#     return np.frombuffer(shared_array, dtype="int32")
#
#
# def process_users(shared_array, num_processors, cnt_processor, user_pos, planes):
#     """
#     :param shared_array:
#     :param num_processors:
#     :param cnt_processor:
#     :param user_pos:
#     :param planes:
#     :return:
#     Divide the users among the processors, does not work well, since startup of processor takes too long
#     """
#     num_users = len(shared_array)
#     set_length = np.floor(num_users / num_processors)
#     for idx_user in range(cnt_processor*set_length,(cnt_processor+1)*set_length):
#         if test_point_within_pyramid(user_pos[idx_user, :], planes):
#             shared_array[idx_user] = 1  # Within swath


@jit(nopython=True)
def check_users_in_plane(user_metric, user_pos, planes, cnt_epoch):
    n_users = len(user_metric)
    for idx_user in range(n_users):
        if test_point_within_pyramid(user_pos[idx_user, :], planes):
            user_metric[idx_user,cnt_epoch] = 1  # Within swath
    return user_metric[:,cnt_epoch]


def det_sza_fast(user_metric, user_pos_lla, epoch, cnt_epoch): # In fact very slow, TBD

    n_users = len(user_metric)
    for idx_user in range(n_users):
        location = EarthLocation(lat=user_pos_lla[idx_user,0]*u.rad,
                                 lon=user_pos_lla[idx_user,1]*u.rad,
                                 height=user_pos_lla[idx_user,2]*u.m)
        altazframe = AltAz(obstime=epoch, location=location)
        sun_altaz = get_sun(epoch).transform_to(altazframe)
        if (user_metric[idx_user,cnt_epoch] > 0):  # Within swath
            user_metric[idx_user, cnt_epoch] = 90 - sun_altaz.alt.value
    return user_metric[:,cnt_epoch]


def det_sza(user_pos_lla, epoch):

    location = EarthLocation(lat=radians(user_pos_lla[0])*u.rad,
                             lon=radians(user_pos_lla[1])*u.rad,
                             height=0*u.m)
    altazframe = AltAz(obstime=epoch, location=location)
    sun_altaz = get_sun(epoch).transform_to(altazframe)
    if (sun_altaz.alt.value > 0):  # Above horizon
        return 90 - sun_altaz.alt.value
    else:
        return 0


@jit(nopython=True)
def check_users_from_nadir(user_metric, user_pos, sat_pos, earth_angle_swath, cnt_epoch):
    """
    :param user_metric: 0 or 1 for covered user[num_users,num_epoch)
    :param user_pos: user position in ECEF [m]
    :param sat_pos: satellite position in ECEF [m]
    :param earth_angle_swath: earth angle swath [rad]
    :param cnt_epoch: current time epoch [-]
    :return: checks if user is within the conical swath
    """
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
    """
    :param v: string
    :return: bool
    Convert string True/False to boolean
    """
    return v.lower() in ("yes", "true", "t", "1")


def comp_gas_attenuation(frequency, elevation):
    """
    :param frequency: frequency [Hz]
    :param elevation: elevation [rad]
    :return: Compute gas attenuation in [dB]
    Compute gas attenuation, Gas attenuation models are given in ITU-R P676-11.
    The model provided is computationally intensive and requires a lot of tabulated data.
    In order to implement this with a low computational burden in a short time, a simplified-statistical model has been derived.
    The elevation dependences have been mapped for each frequency with an exponential fit.
    """

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


def comp_rain_attenuation(frequency, elevation, latitude, height, rain_p_exceed, rainfall_rate, rain_height):
    """
    :param frequency: frequency in [Hz]
    :param elevation: [rad]
    :param latitude: [rad]
    :param height: [m]
    :param rain_p_exceed: [%]
    :param rainfall_rate: [mm/hr]
    :param rain_height: [m]
    :return: Rain attenuation [dB]
    Taken from Gerard Maral, Michel Bousquet. Satellite Communications Systems. John Wiley & Sons. 4th Edition, 2002.
    """

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


def temp_brightness(frequency, elevation):
    """
    :param frequency: [Hz]
    :param elevation:  [rad]
    :return: brightness temp sky [K]
    """

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


def comp_cn0_required(modulation, ber, datarate):
    """
    :param modulation: Modulation type BPSK/QPSK
    :param ber: BER in multiples of 10
    :param datarate: datarate in bps
    :return: CN0 required in dB
    """

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


@jit(nopython=True)
def inverse4by4(a):
    """
    :param a: 4 by 4 numpy array
    :return: inverse of a
    """

    out = np.zeros((4,4))

    tmp = np.zeros(12)
    src = np.zeros(16)
    dst = np.zeros(16)

    # copy input matrix and transpose*/
    src[0]=a[0, 0]
    src[1]=a[1, 0]
    src[2]=a[2, 0]
    src[3]=a[3, 0]
    src[4]=a[0, 1]
    src[5]=a[1, 1]
    src[6]=a[2, 1]
    src[7]=a[3, 1]
    src[8]=a[0, 2]
    src[9]=a[1, 2]
    src[10]=a[2, 2]
    src[11]=a[3, 2]
    src[12]=a[0, 3]
    src[13]=a[1, 3]
    src[14]=a[2, 3]
    src[15]=a[3, 3]

    # calculate pairs for first 8 elements (cofactors) */
    tmp[0] = src[10] * src[15]
    tmp[1] = src[11] * src[14]
    tmp[2] = src[9] * src[15]
    tmp[3] = src[11] * src[13]
    tmp[4] = src[9] * src[14]
    tmp[5] = src[10] * src[13]
    tmp[6] = src[8] * src[15]
    tmp[7] = src[11] * src[12]
    tmp[8] = src[8] * src[14]
    tmp[9] = src[10] * src[12]
    tmp[10] = src[8] * src[13]
    tmp[11] = src[9] * src[12]

    # calculate first 8 elements (cofactors) */
    dst[0] = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7]
    dst[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7]
    dst[1] = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7]
    dst[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7]
    dst[2] = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7]
    dst[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7]
    dst[3] = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6]
    dst[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6]
    dst[4] = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3]
    dst[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3]
    dst[5] = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3]
    dst[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3]
    dst[6] = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3]
    dst[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3]
    dst[7] = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2]
    dst[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2]

    # calculate pairs for second 8 elements (cofactors) */
    tmp[0] = src[2]*src[7]
    tmp[1] = src[3]*src[6]
    tmp[2] = src[1]*src[7]
    tmp[3] = src[3]*src[5]
    tmp[4] = src[1]*src[6]
    tmp[5] = src[2]*src[5]
    tmp[6] = src[0]*src[7]
    tmp[7] = src[3]*src[4]
    tmp[8] = src[0]*src[6]
    tmp[9] = src[2]*src[4]
    tmp[10] = src[0]*src[5]
    tmp[11] = src[1]*src[4]

    # calculate second 8 elements (cofactors) */
    dst[8] = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15]
    dst[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15]
    dst[9] = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15]
    dst[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15]
    dst[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15]
    dst[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15]
    dst[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14]
    dst[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14]
    dst[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9]
    dst[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10]
    dst[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10]
    dst[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8]
    dst[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8]
    dst[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9]
    dst[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9]
    dst[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8]

    # calculate determinant */
    det=src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3]

    # calculate matrix inverse */
    det = 1/det
    for j in range(16):
        dst[j] *= det

    out[0,0]=dst[0]
    out[0,1]=dst[1]
    out[0,2]=dst[2]
    out[0,3]=dst[3]
    out[1,0]=dst[4]
    out[1,1]=dst[5]
    out[1,2]=dst[6]
    out[1,3]=dst[7]
    out[2,0]=dst[8]
    out[2,1]=dst[9]
    out[2,2]=dst[10]
    out[2,3]=dst[11]
    out[3,0]=dst[12]
    out[3,1]=dst[13]
    out[3,2]=dst[14]
    out[3,3]=dst[15]

    return out

@jit(nopython=True)
def ecef2enu(a, user_lat, user_lon):
    """
    :param a: point a from which East North Up is
    :param user_lat: [rad]
    :param user_lon: [rad]
    :return: returns coordinates in a local east-north-up (ENU) Cartesian system,
    corresponding to coordinates X, Y, Z in an Earth-Centered Earth-Fixed (ECEF) spheroid-centric Cartesian system.
    """

    r = np.zeros((3,4))

    r[0, 0] = -np.sin(user_lon)
    r[0, 1] = np.cos(user_lon)
    r[0, 2] = 0
    r[0, 3] = 0

    r[1, 0] = -np.sin(user_lat)*np.cos(user_lon)
    r[1, 1] = -np.sin(user_lat)*np.sin(user_lon)
    r[1, 2] = np.cos(user_lat)
    r[1, 3] = 0

    r[2, 0] = np.cos(user_lat)*np.cos(user_lon)
    r[2, 1] = np.cos(user_lat)*np.sin(user_lon)
    r[2, 2] = np.sin(user_lat)
    r[2, 3] = 0

    q = np.dot(r,a)  # np.matmul not supported by numba
    q = np.dot(q,np.transpose(r))  # np.matmul not supported by numba

    return q


def dish_pattern(frequency, diameter, max_gain, theta):
    """
    :param frequency: in Hz
    :param diameter: in m
    :param max_gain: in dB in boresight
    :param theta off boresight angle: in radians
    :return: gain in dB
    """
    lam = 3e8 / frequency
    k = 2 * PI  # wavelength
    r = diameter / lam / 2  # in units of lambda
    # array factor (E-Field) for a circular antenna
    pattern = lambda theta: (2. * jv(1, k * r * sin(theta)) / (k * r * sin(theta))) ** 2

    return 10.0 * np.log10(pattern(theta)) + max_gain # radiated pattern in dB


def dish_pattern_manual(points, angle_off_boresight):
    """
    :param points: list with points as function of elevation ranging from 0 - 180 deg off boresight in [dB]
    :param angle_off_boresight: angle off boresight in [radians]
    :return: gain at angle given in [dB]
    """

    tck = interpolate.splrep(points[:, 0], points[:, 1], s=0)  # cubic interpolation
    xnew = np.abs(degrees(angle_off_boresight))

    return interpolate.splev(xnew, tck, der=0)


@u.quantity_input(ltan=u.hourangle)
def raan_from_ltan(epoch, ltan=12.0):
    """RAAN angle from LTAN for SSO around the earth
    Parameters
    ----------
    epoch : ~astropy.time.Time
         Value of time to calculate the RAAN for
    ltan: ~astropy.units.Quantity
         Decimal hour between 0 and 24
    Returns
    -------
    RAAN: ~astropy.units.Quantity
        Right ascension of the ascending node angle in GCRS
    Note
    ----
    Calculations of the sun mean longitude and equation of time
    follow "Fundamentals of Astrodynamics and Applications"
    Fourth edition by Vallado, David A.
    """
    constants_J2000 = time.Time("J2000", scale="tt")
    T_UT1 = ((epoch.ut1 - constants_J2000).value / 36525.0) * u.deg
    T_TDB = ((epoch.tdb - constants_J2000).value / 36525.0) * u.deg

    # Apparent sun position
    sun_position = coordinates.get_sun(epoch)

    # Calculate the sun apparent local time
    salt = sun_position.ra + 12 * u.hourangle

    # Use the equation of time to calculate the mean sun local time (fictional sun without anomalies)

    # sun mean anomaly
    M_sun = 357.5291092 * u.deg + 35999.05034 * T_TDB

    # sun mean longitude
    l_sun = 280.460 * u.deg + 36000.771 * T_UT1
    l_ecliptic_part2 = 1.914666471 * u.deg * np.sin(
        M_sun
    ) + 0.019994643 * u.deg * np.sin(2 * M_sun)
    l_ecliptic = l_sun + l_ecliptic_part2

    eq_time = (
        -l_ecliptic_part2
        + 2.466 * u.deg * np.sin(2 * l_ecliptic)
        - 0.0053 * u.deg * np.sin(4 * l_ecliptic)
    )

    # Calculate sun mean local time

    smlt = salt + eq_time

    # Desired angle between sun and ascending node
    alpha = (coordinates.Angle(ltan).wrap_at(24 * u.hourangle)).to(u.rad)

    # Use the mean sun local time calculate needed RAAN for given LTAN
    raan = smlt + alpha
    return raan