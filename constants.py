import math

PI = math.pi
RAD = PI / 180.0 / 3600  # Radians per arcseconds of degrees
RAD2 = PI / 180.0  # Radians per degree

# Nutation
N_COEFF_IAU_80 = 106

# General
MJD_J2000 = 51544.5  # Modif. Julian Date of J2000.0
JD_J2000 = 2451545.0  # Julian Date of J2000.0

G = 6.673e-11  # Gravitational constant [m^3/(kg.s^2)]McCarthy (1996)
g = 9.7803278  # Mean equatorial gravity [m.s-2]MacCarthy(2004)
AU = 149597870691.0  # Astronomical unit [m]DE405 (Standish 1998)
A_MOON = 384400.0e3  # Moon major axis [m]Seidelmann 1992
C_LIGHT = 299792458.0  # Speed of light  [m/s]IAU 1976 (Seidelmann 1992)

K_BOLTZMANN = -228.6  # dBJ/K

R_SUN = 6.9599e8  # Radius Sun [m] Seidelmann 1992
R_EARTH = 6378136.3  # Radius Earth [m] EGM2008 & EGM96
R_MOON = 1738.200e3

omega_Earth = 7.29211514670698e-5  # [rad/s]Aoki 1982, NIMA 1997  Earth rotation

# Gravitational coefficientS
GM_EARTH = 398600.4415e+9  # [m^3/s^2]JGM3
GM_SUN = 1.327122000000e+20
GM_MOON = 4902.802953597e+9  # [m^3/s^2]DE405 (Standish 1998)
GM_MERCURY = 2.203209000e+13
GM_VENUS = 3.248585920790e+14
GM_MARS = 4.282837190128e+13
GM_JUPITER = 1.267127678578e+17
GM_SATURN = 3.794062606114e+16
GM_URANUS = 5.8032000e+15
GM_NEPTUNE = 6.836534062383e+15

M = GM_EARTH / G

SOLAR_FLUX = 1367.0  # Solar flux and solar radiation pressure at 1 AU
SOLAR_PRESSURE = SOLAR_FLUX / C_LIGHT  # [N/m^2] (~1367 W/m^2) IERS 96

ALPHA = 0.3  # Albedo

RHO = 1025.0  # Ocean Tide Density of seawater [kg.m-3] MacCarthy(2004)
CONST_OCEAN_TIDE = (4 * PI * G * RHO) / g  # Constant used in the ocean tides corrections (p.67 McCarthy (2004))
