#### Author of this code: James Kirk with contributions from James McCormac
#### Contact: jameskirk@live.co.uk 

from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import argparse
from astropy.coordinates import EarthLocation,SkyCoord

parser = argparse.ArgumentParser(description='Convert times for a target from MJD to BJD and HJD. This is code written by James McCormac. Note that the observatory must be correctly defined!')
parser.add_argument('mjd',type=float,help='Time of mid-transit in MJD')
parser.add_argument('--c',help="Coords of target, to be given as --c 'XX:XX:XX.XX XX:XX:XX.X'",type=str,nargs='+')
parser.add_argument('--obs',help="Define the observatory where the data were taken. Default = 'Roque de los Muchachos'",default="Roque de los Muchachos")
args = parser.parse_args()

# Observatory
OBSERVATORY = EarthLocation.of_site(args.obs)
# print(OBSERVATORY)


def getLightTravelTimes(ra, dec, time_to_correct):
    """
    Get the light travel times to the helio- and
    barycentres
    Parameters
    ----------
    ra : str
        The Right Ascension of the target in hourangle
        e.g. 16:00:00
    dec : str
        The Declination of the target in degrees
        e.g. +20:00:00
    time_to_correct : astropy.Time object
        The time of observation to correct. The astropy.Time
        object must have been initialised with an EarthLocation
    Returns
    -------
    ltt_bary : TimeDelta
        The light travel time to the barycentre
    ltt_helio : TimeDelta
        The light travel time to the heliocentre
    Raises
    ------
    None
    """
    target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ltt_bary = time_to_correct.light_travel_time(target)
    ltt_helio = time_to_correct.light_travel_time(target, 'heliocentric')
    return ltt_bary, ltt_helio



time_jd = Time(args.mjd, format='mjd',scale='utc', location=OBSERVATORY) # gp


ra = args.c[0].split()[0]
dec = args.c[0].split()[1]
ltt_bary, ltt_helio = getLightTravelTimes(ra, dec, time_jd)
time_bary = time_jd.tdb + ltt_bary
time_helio = time_jd.utc + ltt_helio

# print the results
print("Barycentric Julian Date (TDB): {0:.8f}".format(time_bary.jd))
print("Heliocentric Julian Date (UTC): {0:.8f}".format(time_helio.jd))


### Additional useful code...

# def helio_to_bary(coords, hjd, obs_name):
#     helio = Time(hjd, scale='utc', format='jd')
#     obs = EarthLocation.of_site(obs_name)
#     star = SkyCoord(coords, unit=(u.hour, u.deg))
#     ltt = helio.light_travel_time(star, 'heliocentric', location=obs)
#     guess = helio - ltt
#     # if we assume guess is correct - how far is heliocentric time away from true value?
#     delta = (guess + guess.light_travel_time(star, 'heliocentric', obs)).jd  - helio.jd
#     # apply this correction
#     guess -= delta * u.d
#
#     ltt = guess.light_travel_time(star, 'barycentric', obs)
#     return guess.tdb + ltt
#
#
# def bary_to_helio(coords, bjd, obs_name):
#     bary = Time(bjd, scale='tdb', format='jd')
#     obs = EarthLocation.of_site(obs_name)
#     star = SkyCoord(coords, unit=(u.hour, u.deg))
#     ltt = bary.light_travel_time(star, 'barycentric', location=obs)
#     guess = bary - ltt
#     delta = (guess + guess.light_travel_time(star, 'barycentric', obs)).jd  - bary.jd
#     guess -= delta * u.d
#
#     ltt = guess.light_travel_time(star, 'heliocentric', obs)
#     return guess.utc + ltt
