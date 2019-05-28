import math
from math import sin, cos, atan2, atan, fabs, acos
import numpy as np
from astropy.coordinates import EarthLocation
from math import tan, sqrt, asin, degrees, radians
from numpy import dot, arccos, clip, cross, sin, cos
from numpy.linalg import norm
# Modules from project
from constants import R_EARTH, PI, GM_EARTH


class Plane:  # Definition of a plane class to be used in OBS analysis
    def __init__(self, a, b, c):
        # self.n = normalize(cross(b - a, c - a))  # n is plane normal. Point X on the plane satisfies Dot(n, X) = d
        # n is plane normal. Point X on the plane satisfies Dot(n, X) = d. Do not need to normalize here.
        self.n = cross(b - a, c - a)
        # self.d = dot(self.n, a)  # d = dot(n, p)  # distance plane to origin, in this case always 0


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
def lla2xyz(lla):

    xyz = 3*[0.0]
    
    a = 6378137.0000
    b = 6356752.3142
    e = sqrt(1 - pow((b / a), 2))
    
    sin_phi = sin(lla[0])
    cos_phi = cos(lla[0])
    cos_lam = cos(lla[1])
    sin_lam = sin(lla[1])
    tan2phi = pow((tan(lla[0])), 2.0)
    tmp = 1.0 - e*e
    tmp_den = sqrt(1.0 + tmp * tan2phi)
    
    xyz[0] = (a * cos_lam) / tmp_den + lla[2] * cos_lam*cos_phi
    
    xyz[1] = (a * sin_lam) / tmp_den + lla[2] * sin_lam*cos_phi
    
    tmp2 = sqrt(1 - e * e * sin_phi * sin_phi)
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
def xyz2lla(xyz):

    x2 = xyz[0] * xyz[0]
    y2 = xyz[1] * xyz[1]
    z2 = xyz[2] * xyz[2]
    
    sma = 6378137.0000  # earth radius in meters
    smb = 6356752.3142  # earth semi minor in meters
    
    e = sqrt(1.0 - (smb / sma)*(smb / sma))
    b2 = smb*smb
    e2 = e*e
    ep = e * (sma / smb)
    r = sqrt(x2 + y2)
    r2 = r*r
    E2 = sma * sma - smb*smb
    big_f = 54.0 * b2*z2
    G = r2 + (1.0 - e * e) * z2 - e * e*E2
    small_c = (e * e * e * e * big_f * r2) / (G * G * G)
    s = pow(1.0 + small_c + sqrt(small_c * small_c + 2.0 * small_c), 1.0 / 3.0)
    P = big_f / (3.0 * (s + 1.0 / s + 1.0)*(s + 1.0 / s + 1.0) * G * G)
    Q = sqrt(1 + 2 * e2 * e2 * P)
    ro = -(P * e * e * r) / (1.0 + Q) + sqrt((sma * sma / 2.0)*(1.0 + 1.0 / Q) -
                                             (P * (1 - e * e) * z2) / (Q * (1.0 + Q)) - P * r2 / 2.0)
    tmp = (r - e * e * ro)*(r - e * e * ro)
    U = sqrt(tmp + z2)
    V = sqrt(tmp + (1 - e * e) * z2)
    zo = (b2 * xyz[2]) / (sma * V)
    
    height = U * (1 - b2 / (sma * V))
    
    lat = atan((xyz[2] + ep * ep * zo) / r)
    
    temp = atan(xyz[1] / xyz[0])
    
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
def kep2xyz(mjd_requested, kepler):

    out = 6*[0.0]

    # Compute radius, corrected right ascension and mean anomaly
    time_from_ref = (mjd_requested - kepler.epoch_mjd) * 86400
    mean_motion = sqrt(GM_EARTH / (pow(kepler.semi_major_axis, 3)))  #radians/s
    mean_anomaly = kepler.mean_anomaly + (mean_motion * time_from_ref)

    # if eccentricity equals zero eccentric anomaly is equal mean anomaly
    if kepler.eccentricity == 0:
        eccentric_anomaly = mean_anomaly
    else:
        eccentric_anomaly = newton_raphson(mean_anomaly, kepler.eccentricity)

    radius = kepler.semi_major_axis * (1 - kepler.eccentricity * cos(eccentric_anomaly))

    sin_nu_k = ((sqrt(1 - kepler.eccentricity * kepler.eccentricity)) * sin(eccentric_anomaly)) / (1 - kepler.eccentricity * cos(eccentric_anomaly))
    cos_nu_k = (cos(eccentric_anomaly) - kepler.eccentricity) / (1 - kepler.eccentricity * cos(eccentric_anomaly))
    true_anomaly = atan2(sin_nu_k, cos_nu_k)
    arg_of_lat = kepler.arg_perigee + true_anomaly

    # Apply rotation from elements to xyz
    xyk1 = radius * cos(arg_of_lat)
    xyk2 = radius * sin(arg_of_lat)
    out[0] = cos(kepler.right_ascension) * xyk1 + -cos(kepler.inclination) * sin(kepler.right_ascension) * xyk2
    out[1] = sin(kepler.right_ascension) * xyk1 + cos(kepler.inclination) * cos(kepler.right_ascension) * xyk2
    out[2] = sin(kepler.inclination) * xyk2

    # Compute velocity components
    factor = sqrt(GM_EARTH / kepler.semi_major_axis / (1 - pow(kepler.eccentricity, 2)))
    sin_w = sin(kepler.arg_perigee)
    cos_w = cos(kepler.arg_perigee)
    sin_o = sin(kepler.right_ascension)
    cos_o = cos(kepler.right_ascension)
    sin_i = sin(kepler.inclination)
    cos_i = cos(kepler.inclination)
    sin_t = sin(true_anomaly)
    cos_t = cos(true_anomaly)
    l1 = cos_w * cos_o - sin_w * sin_o*cos_i
    m1 = cos_w * sin_o + sin_w * cos_o*cos_i
    n1 = sin_w*sin_i
    l2 = -sin_w * cos_o - cos_w * sin_o*cos_i
    m2 = -sin_w * sin_o + cos_w * cos_o*cos_i
    n2 = cos_w*sin_i

    out[3] = factor * (-l1 * sin_t + l2 * (kepler.eccentricity + cos_t))
    out[4] = factor * (-m1 * sin_t + m2 * (kepler.eccentricity + cos_t))
    out[5] = factor * (-n1 * sin_t + n2 * (kepler.eccentricity + cos_t))

    return out


# Iterates to solution of eccentric_anomaly using numerical solution
# Only necessary if eccentricity is unequal to 0
# 
# Ref. http://en.wikipedia.org/wiki/Newton's_method
# 
# @param MeanAnomaly mean anomaly (radians)
# @param Eccentricity eccentricity (-)
# @return  Eccentric anomaly (radians)
def newton_raphson (mean_anomaly, eccentricity):
    k = 0
    big_e = 50*[0.0]

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
def spin_vector(angle, vector):

    out = 6*[0.0]
    
    # Compute angles to save time
    cosst = cos(angle)
    sinst = sin(angle)
    
    out[0] = cosst * vector[0] - sinst * vector[1]
    out[1] = sinst * vector[0] + cosst * vector[1]
    out[2] = vector[2]
    
    out[3] = cosst * vector[3] - sinst * vector[4]
    out[4] = sinst * vector[3] + cosst * vector[4]
    out[5] = vector[5]
    
    return out


# Compute Azimuth and Elevation from User to Satellite assuming the Earth is a perfect sphere
#
# @param Xs Satellite position in ECEF (m/double)
# @param Xu User position in ECEF (m/double)
# @param AzEl Azimuth [0] and Elevation [1] (radians/double)
def calc_az_el(xs, xu):

    az_el = 2*[0.0]

    e3by3 = [[0,0,0],[0,0,0],[0,0,0]]
    d = 3*[0.0]

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
def line_sphere_intersect(x1, x2, sphere_radius, sphere_center):

    intersect = True

    i_x1 = 3*[0.0]
    i_x2 = 3*[0.0]

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


def dist_point_plane(point_q, plane_p):  # not really distance but if point is more than 90 deg away from normal
    return dot(point_q, plane_p.n)  # - plane_p.d


def test_point_within_pyramid(point_q, planes_h):
    for i in range(4):
        if dist_point_plane(point_q, planes_h[i]) > 0.0:  # on bad side of plane, faster implementation
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
    # Thanks user MNKY at http://math.stackexchange.com/a/1432182/81266
    proj = x_project_on_v(a, b)
    w = cross(b, proj['perp'])
    return proj['par'] + proj['perp'] * cos(theta) + norm(proj['perp']) * make_unit(w) * sin(theta)


def det_swath_radius(H, inc_angle): # altitude in [m], incidence angle alfa in [radians]
    A = (1 / tan(inc_angle) / tan(inc_angle) + 1)
    B = -2*(H+R_EARTH)
    C = (H+R_EARTH) * (H+R_EARTH) - R_EARTH * R_EARTH / tan(inc_angle) / tan(inc_angle)
    det = sqrt(B*B-4*A*C)
    y1 = (-B + det)/2/A
    x = sqrt(R_EARTH*R_EARTH - y1*y1)
    return x # in [m]


def det_oza(H, inc_angle):  # altitude in [m], incidence angle  in [radians]
    x = det_swath_radius(H, inc_angle)
    beta = degrees(asin(x / R_EARTH))
    ia = 90 - degrees(inc_angle) - beta
    oza = 90 - ia
    return oza  # in [deg]


def det_oza_fast(H, R_EARTH, inc_angle):  # altitude and R_EARTH in [m], incidence angle  in [radians]
    oza = asin((R_EARTH+H)/R_EARTH*sin(inc_angle))
    return oza


def earth_angle_beta_deg(x):
    # Returns the Earth beta angle in degrees
    beta = degrees(asin(x / R_EARTH))
    return beta


def angle_two_vectors(u, v, norm_u, norm_v):
    # Returns angle in [radians]
    # Pre computed the norm of the second vector
    c = dot(u, v) / norm_u / norm_v
    return arccos(clip(c, -1, 1))


# Compute numerically the solution to the incidence angle alfa from a swath width from nadir
# swath width in [m]
# radius Earth in [m]
# altitude satellite in [m]
# return incidence angle in [rad]
def incl_from_swath(swath_width, radius_earth, altitude):
    solution = atan(sin(swath_width / radius_earth) /
                    ((radius_earth + altitude) / radius_earth - cos(swath_width / radius_earth)))
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

# Convert string True/False to boolean
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

