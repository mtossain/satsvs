import math
from math import sin, cos, tan, atan2, atan, floor, sqrt, fabs
import constants
from constants import M_PI, GM_Earth


"""**
* Compute Modified Julian Date from YYYY and Day Of Year pair
* Time functions from SP3 library B.Remondi.
*
* @param YYYY Year (year)
* @param DOY Day of year (days)
* @return Modified Julian Date
*"""


def YYYYDOY2MJD(YYYY, DOY):

    mjd_jan1_1901 = 15385
    days_per_second = 0.00001157407407407407

    yday = int(math.floor(DOY))
    fDOY = DOY - yday
    full_seconds = fDOY * 86400.0
    hours = full_seconds / 3600.0
    hour = int(math.floor(hours))
    rest = full_seconds - hour * 3600
    minutes = rest / 60.0
    minute = int(math.floor(minutes))
    second = full_seconds - hour * 3600 - minute * 60

    years_into_election = (YYYY - 1) % 4
    elections = (YYYY - 1901) / 4

    pfmjd = (hour * 3600 + minute * 60 + second) * days_per_second
    pmjd = mjd_jan1_1901 + elections * 1461 + years_into_election * 365 + yday - 1

    return pfmjd + pmjd

# Compute Greenwich Mean Sidereal Time from Modified Julian Date
# Ref. Time functions from SP3 library B.Remondi.
#
# @param MJD_Requested Modified Julian Date (days/double)
# @return Greenwich Mean Sidereal Time (radians/double) [0 to 2pi]

def MJD2GMST(MJD_Requested):

    JD = MJD_Requested + 2400000.5

    # centuries elapsed since 2000,1,1.5
    TU = (JD - 2451545) / 36525

    # gmst in time-seconds at 0 ut
    THETAN = 24110.54841 + (8640184.812866 * TU)+(0.093104 * TU * TU)-(6.2e-6 * TU * TU * TU)

    # correction for other time than 0 ut modulo rotations
    THETA0 = 43200.0 + (3155760000.0 * TU)
    THETAN = THETAN + (THETA0 % 86400)

    # gmst in radians
    THETAN = ((THETAN / 43200) * M_PI) % (2 * M_PI)
    if THETAN < 0:
        THETAN = THETAN + (2 * M_PI)

    return THETAN

# Convert from latitude, longitude and height to ECF cartesian coordinates, taking
# into account the Earth flattening
#
# Ref. Understanding GPS: Principles and Applications,
# Elliott D. Kaplan, Editor, Artech House Publishers, Boston 1996.
#
# @param LLA Latitude rad/[-pi/2,pi/2] positive N, Longitude rad/[-pi,pi] positive E, height above ellipsoid
# @param XYZ ECEF XYZ coordinates in m


def LLA2XYZ(LLA):

    XYZ = [0,0,0]
    
    a = 6378137.0000
    b = 6356752.3142
    e = sqrt(1 - pow((b / a), 2))
    
    sinphi = sin(LLA[0])
    cosphi = cos(LLA[0])
    coslam = cos(LLA[1])
    sinlam = sin(LLA[1])
    tan2phi = pow((tan(LLA[0])), 2.0)
    tmp = 1.0 - e*e
    tmpden = sqrt(1.0 + tmp * tan2phi)
    
    XYZ[0] = (a * coslam) / tmpden + LLA[2] * coslam*cosphi
    
    XYZ[1] = (a * sinlam) / tmpden + LLA[2] * sinlam*cosphi
    
    tmp2 = sqrt(1 - e * e * sinphi * sinphi)
    XYZ[2] = (a * tmp * sinphi) / tmp2 + LLA[2] * sinphi
    
    return XYZ


# Convert from ECF cartesian coordinates to latitude, longitude and height, taking
# into account the Earth flattening
#
# Ref. Converts from WGS-84 X, Y and Z rectangular co-ordinates to latitude,
#      longitude and height. For more information: Simo H. Laurila,
#      "Electronic Surveying and Navigation", John Wiley & Sons (1976).
#
# @param XYZ ECEF XYZ coordinates in m
# @param LLA Latitude rad/[-pi/2,pi/2] positive N, Longitude rad/[-pi,pi] positive E, height above ellipsoid


def XYZ2LLA(XYZ):

    x2 = XYZ[0] * XYZ[0]
    y2 = XYZ[1] * XYZ[1]
    z2 = XYZ[2] * XYZ[2]
    
    sma = 6378137.0000 # earth radius in meters
    smb = 6356752.3142 # earth semi minor in meters
    
    e = sqrt(1.0 - (smb / sma)*(smb / sma))
    b2 = smb*smb
    e2 = e*e
    ep = e * (sma / smb)
    r = sqrt(x2 + y2)
    r2 = r*r
    E2 = sma * sma - smb*smb
    bigf = 54.0 * b2*z2
    G = r2 + (1.0 - e * e) * z2 - e * e*E2
    smallc = (e * e * e * e * bigf * r2) / (G * G * G)
    s = pow(1.0 + smallc + sqrt(smallc * smallc + 2.0 * smallc), 1.0 / 3.0)
    P = bigf / (3.0 * (s + 1.0 / s + 1.0)*(s + 1.0 / s + 1.0) * G * G)
    Q = sqrt(1 + 2 * e2 * e2 * P)
    ro = -(P * e * e * r) / (1.0 + Q) + sqrt((sma * sma / 2.0)*(1.0 + 1.0 / Q)- (P * (1 - e * e) * z2) / (Q * (1.0 + Q)) - P * r2 / 2.0)
    tmp = (r - e * e * ro)*(r - e * e * ro)
    U = sqrt(tmp + z2)
    V = sqrt(tmp + (1 - e * e) * z2)
    zo = (b2 * XYZ[2]) / (sma * V)
    
    height = U * (1 - b2 / (sma * V))
    
    lat = atan((XYZ[2] + ep * ep * zo) / r)
    
    temp = atan(XYZ[1] / XYZ[0])
    
    if XYZ[0] >= 0:
        lon = temp
    elif XYZ[0] < 0 and XYZ[1] >= 0:
        lon = M_PI + temp
    else:
        lon = temp - M_PI
    
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

def KEP2XYZ(MJD_Requested, Kepler):

    Out = [0,0,0,0,0,0]

    # Set constants
    mu = GM_Earth

    # Compute radius, corrected right ascension and mean anomaly
    time_from_ref = (MJD_Requested - Kepler.EpochMJD)*86400
    mean_motion = sqrt(mu / (pow(Kepler.SemiMajorAxis, 3))) #radians/s
    mean_anomaly = Kepler.MeanAnomaly + (mean_motion * time_from_ref)

    # if eccentricity equals zero eccentric anomaly is equal mean anomaly
    if Kepler.Eccentricity == 0:
        eccentric_anomaly = mean_anomaly
    else:
        eccentric_anomaly = NewtonRaphson(mean_anomaly, Kepler.Eccentricity)

    radius = Kepler.SemiMajorAxis * (1 - Kepler.Eccentricity * cos(eccentric_anomaly))

    sin_nu_k = ((sqrt(1 - Kepler.Eccentricity * Kepler.Eccentricity)) * sin(eccentric_anomaly)) / (1 - Kepler.Eccentricity * cos(eccentric_anomaly))
    cos_nu_k = (cos(eccentric_anomaly) - Kepler.Eccentricity) / (1 - Kepler.Eccentricity * cos(eccentric_anomaly))
    true_anomaly = atan2(sin_nu_k, cos_nu_k)
    arg_of_lat = Kepler.ArgOfPerigee + true_anomaly

    # Apply rotation from elements to xyz
    xyk1 = radius * cos(arg_of_lat)
    xyk2 = radius * sin(arg_of_lat)
    Out[0] = cos(Kepler.RAAN) * xyk1 + -cos(Kepler.Inclination) * sin(Kepler.RAAN) * xyk2
    Out[1] = sin(Kepler.RAAN) * xyk1 + cos(Kepler.Inclination) * cos(Kepler.RAAN) * xyk2
    Out[2] = sin(Kepler.Inclination) * xyk2

    # Compute velocity components
    factor = sqrt(mu / Kepler.SemiMajorAxis / (1 - pow(Kepler.Eccentricity, 2)))
    sinw = sin(Kepler.ArgOfPerigee)
    cosw = cos(Kepler.ArgOfPerigee)
    sino = sin(Kepler.RAAN)
    coso = cos(Kepler.RAAN)
    sini = sin(Kepler.Inclination)
    cosi = cos(Kepler.Inclination)
    sint = sin(true_anomaly)
    cost = cos(true_anomaly)
    l1 = cosw * coso - sinw * sino*cosi
    m1 = cosw * sino + sinw * coso*cosi
    n1 = sinw*sini
    l2 = -sinw * coso - cosw * sino*cosi
    m2 = -sinw * sino + cosw * coso*cosi
    n2 = cosw*sini

    Out[3] = factor * (-l1 * sint + l2 * (Kepler.Eccentricity + cost))
    Out[4] = factor * (-m1 * sint + m2 * (Kepler.Eccentricity + cost))
    Out[5] = factor * (-n1 * sint + n2 * (Kepler.Eccentricity + cost))

    return Out


# Iterates to solution of eccentric_anomaly using numerical solution
# Only necessary if eccentricity is unequal to 0
# 
# Ref. http://en.wikipedia.org/wiki/Newton's_method
# 
# @param MeanAnomaly mean anomaly (radians)
# @param Eccentricity eccentricity (-)
# @return  Eccentric anomaly (radians)


def NewtonRaphson (MeanAnomaly, Eccentricity):
    k = 0
    E = 50*[0]

    E[1] = MeanAnomaly
    E[0] = MeanAnomaly * 2
    while fabs(E[k + 1] / E[k] - 1) > 1e-15:
        k += 1
        E[k + 1] = E[k]-((E[k] - Eccentricity * sin(E[k]) - MeanAnomaly) / (1 - Eccentricity * cos(E[k])))

    return E[k + 1]


# Rotate vector on z-axis
#
# @param Angle Rotation angle (radians/double)
# @param Vector Position input vector (1:6) (meters/double)
# @param Out Rotated position vector (1:6) (meters/double)

def SpinVector(Angle, Vector):

    Out = [0,0,0,0,0,0]
    
    # Compute angles to save time
    cosst = cos(Angle)
    sinst = sin(Angle)
    
    Out[0] = cosst * Vector[0] - sinst * Vector[1]
    Out[1] = sinst * Vector[0] + cosst * Vector[1]
    Out[2] = Vector[2]
    
    Out[3] = cosst * Vector[3] - sinst * Vector[4]
    Out[4] = sinst * Vector[3] + cosst * Vector[4]
    Out[5] = Vector[5]
    
    return Out

# Compute Azimuth and Elevation from User to Satellite assuming the Earth is a perfect sphere
#
# @param Xs Satellite position in ECEF (m/double)
# @param Xu User position in ECEF (m/double)
# @param AzEl Azimuth [0] and Elevation [1] (radians/double)


def CalcAzEl(Xs, Xu):

    AzEl=[0,0]

    e = [[0,0,0],[0,0,0],[0,0,0]]
    d = [0,0,0]

    x = Xu[0]
    y = Xu[1]
    z = Xu[2]
    p = sqrt(x * x + y * y)

    R = sqrt(x * x + y * y + z * z)
    e[0][0] = -(y / p)
    e[0][1] = x / p
    e[0][2] = 0.0
    e[1][0] = -(x * z / (p * R))
    e[1][1] = -(y * z / (p * R))
    e[1][2] = p / R
    e[2][0] = x / R
    e[2][1] = y / R
    e[2][2] = z / R

    # matrix multiply vector from user */
    for k in range(3):
        d[k] = 0.0
        for i in range (3):
            d[k] += (Xs[i] - Xu[i]) * e[k][i]

    s = d[2] / sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if s == 1.0:
        AzEl[1] = 0.5 * M_PI
    else:
        AzEl[1] = atan(s / sqrt(1.0 - s * s))

    if d[1] == 0.0 and d[0] > 0.0:
        AzEl[0] = 0.5 * M_PI
    elif d[1] == 0.0 and d[0] < 0.0:
        AzEl[0] = 1.5 * M_PI
    else:
        AzEl[0] = atan(d[0] / d[1])
        if d[1] < 0.0:
            AzEl[0] += M_PI
        elif d[1] > 0.0 and d[0] < 0.0:
            AzEl[0] += 2.0 * M_PI

    return AzEl[0], AzEl[1]

# Computes two intersection points for a line and a sphere
# Returns false if the line does not intersect the sphere
# Returns two equal intersection points if the line is tangent to the sphere
# Returns two different intersection points if the line gos through the sphere
# Ref: GSSF v2.1
#
# @param X1 Point 1 on the line
# @param X2 Point 2 on the line
# @param SphereRadius Radius of the sphere
# @param SphereCenterX Center point of the sphere
# @param iX1 Intersection point one
# @param iX2 Intersection point two
# @return Returns false if the line does not intersect the sphere

def GetLineSphereIntersection(X1, X2, SphereRadius, SphereCenterX) :

    iX1 = [0,0,0]
    iX2 = [0,0,0]

    a = (X2[0] - X1[0])*(X2[0] - X1[0]) + (X2[1] - X1[1])*(X2[1] - X1[1]) + (X2[2] - X1[2])*(X2[2] - X1[2])

    b = 2.0 * ((X2[0] - X1[0])*(X1[0] - SphereCenterX[0]) +(X2[1] - X1[1])*(X1[1] - SphereCenterX[1]) + \
                      (X2[2] - X1[2])*(X1[2] - SphereCenterX[2]))

    c = SphereCenterX[0] * SphereCenterX[0] + SphereCenterX[1] * SphereCenterX[1] + \
        SphereCenterX[2] * SphereCenterX[2] + X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2] - \
        2.0 * (SphereCenterX[0] * X1[0] + SphereCenterX[1] * X1[1] + SphereCenterX[2] * X1[2]) - \
        SphereRadius*SphereRadius

    inSqrt = (b * b - 4.0 * a * c)

    #No intersection
    if inSqrt < 0:
        Intersect = False

    # Tangent (0) or intersection (+)
    u1 = (-b + sqrt(inSqrt)) / 2.0 / a
    u2 = (-b - sqrt(inSqrt)) / 2.0 / a

    # Intersection points
    iX1[0] = X1[0] + u1 * (X2[0] - X1[0])
    iX1[1] = X1[1] + u1 * (X2[1] - X1[1])
    iX1[2] = X1[2] + u1 * (X2[2] - X1[2])

    iX2[0] = X1[0] + u2 * (X2[0] - X1[0])
    iX2[1] = X1[1] + u2 * (X2[1] - X1[1])
    iX2[2] = X1[2] + u2 * (X2[2] - X1[2])

    return Intersect,iX1,iX2


# Returns the geometry matrix from user to satellite (in local ENU frame rather than
# observation matrix in ECEF frame)
#
# Ref. http://en.wikipedia.org/wiki/Newton's_method
#
# @param User Object with user values (see segments.h)
# @param User2SatelliteList Array with Ground2SpaceLink objects (see segments.h)
# @param iUser Index of user to UserList & User2SatelliteList (-/integer)
#
# @return Geometry matrix nx4 with n equal to number of satellites in view

def User2SatGeometryMatrix (UserList, Ground2SpaceLink, User2SatelliteList, iUser):

    G = UserList[iUser].NumSatInView*[[0,0,0,0]]

    NumSat = Ground2SpaceLink.NumSat; # ??? is this true??? should it not be user2space link?

    for i in range(UserList[iUser].NumSatInView):  #Loop satellites in view
        el = User2SatelliteList[iUser * NumSat + UserList[iUser].IdxSatInView[i]].Elevation
        az = User2SatelliteList[iUser * NumSat + UserList[iUser].IdxSatInView[i]].Azimuth
        G[i][0] = -cos(el) * sin(az)
        G[i][1] = -cos(el) * cos(az)
        G[i][2] = -sin(el)
        G[i][3] = 1.0

    return G
