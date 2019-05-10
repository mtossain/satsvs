import math

pi = math.pi
Rad = (pi / (180.0)) / 3600  # Radians per arcseconds of degrees
Rad2 = pi / (180.0)  # Radians per degree

# Nutation
N_coeffIAU80 = 106

# General
MJD_J2000 = 51544.5  # Modif. Julian Date of J2000.0
JD_J2000 = 2451545.0  # Julian Date of J2000.0

G = 6.673e-11  # Gravitational constant [m^3/(kg.s^2)]McCarthy (1996)
g = 9.7803278  # Mean equatorial gravity [m.s-2]MacCarthy(2004)
AU = 149597870691.0  # Astronomical unit [m]DE405 (Standish 1998)
a_moon = 384400.0e3  # Moon major axis [m]Seidelmann 1992
c_light = 299792458.0  # Speed of light  [m/s]IAU 1976 (Seidelmann 1992)

k_boltzmann = -228.6  # dBJ/K

R_Sun = 6.9599e8  # Radius Sun [m] Seidelmann 1992
R_Earth = 6378136.3  # Radius Earth [m] EGM2008 & EGM96
R_Moon = 1738.200e3

omega_Earth = 7.29211514670698e-5  # [rad/s]Aoki 1982, NIMA 1997  Earth rotation

# Gravitational coefficient
GM_Earth = 398600.4415e+9  # [m^3/s^2]JGM3
GM_Sun = 1.327122000000e+20
GM_Moon = 4902.802953597e+9  # [m^3/s^2]DE405 (Standish 1998)
GM_Mercury = 2.203209000e+13
GM_Venus = 3.248585920790e+14
GM_Mars = 4.282837190128e+13
GM_Jupiter = 1.267127678578e+17
GM_Saturn = 3.794062606114e+16
GM_Uranus = 5.8032000e+15
GM_Neptune = 6.836534062383e+15

M = GM_Earth / G

# Solar flux and solar radiation pressure at 1 AU
solar_flux = 1367.0
solar_pressure = solar_flux / c_light  # [N/m^2] (~1367 W/m^2)IERS 96

# Albedo
alpha = 0.3

# Ocean Tides
# Density of seawater
Rho = 1025.0  # [kg.m-3] MacCarthy(2004)
# Constant used in the ocean tides corrections (p.67 McCarthy (2004))
CONST = (4 * pi * G * Rho) / g
