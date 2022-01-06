import ephem
import os
import subprocess
import pytz
from timezonefinder import TimezoneFinder

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


class TuvSettingsVO():
    """
    This class mimics the input table for TUV
    """
    inpfil = 'usrinp'
    outfil = None
    nstr = None
    lon = None
    lat = None
    pass


def get_tuv_settings():
    if 'TUV_SCRATCH' not in os.environ.keys():  # where to run TUV
        raise Exception('TUV utils',' please configure TUV_SCRATCH variable in bash')
    if 'TUV_SOURCE' not in os.environ.keys():  # where is the TUV source (and executable)
        raise Exception('TUV utils', ' please configure TUV_SOURCE variable in bash')

    return os.environ['TUV_SCRATCH'], os.environ['TUV_SOURCE']


def prep_ruv_environment(tuv_run_dir, tuv_source_dir):
    if not os.path.exists(tuv_run_dir):
        os.makedirs(tuv_run_dir)
        os.mkdir(tuv_run_dir+'INPUTS')
        os.symlink(tuv_source_dir + '/DATAE1/', tuv_run_dir + '/DATAE1')
        os.symlink(tuv_source_dir + '/DATAJ1/', tuv_run_dir + '/DATAJ1')
        os.symlink(tuv_source_dir + '/DATAS1/', tuv_run_dir + '/DATAS1')
        os.symlink(tuv_source_dir + '/tuv', tuv_run_dir + '/tuv')


def update_tuv_input_file(tuv_vo, tuv_input_template_fp, tuv_input_fp):
    # update the TUV input file
    with open(tuv_input_template_fp, 'r+') as file:
        settings = file.readlines()

    # update the settings
    if len(tuv_vo.inpfil) > 6: raise Exception('tuv_vo', 'inpfil is too long')
    if len(tuv_vo.outfil) > 6: raise Exception('tuv_vo', 'inpfil is too long')

    settings[2] = 'inpfil =      {:6s}   outfil =      {:6s}   nstr =   {:11d}\n'.format(tuv_vo.inpfil, tuv_vo.outfil, tuv_vo.nstr)
    settings[3] = 'lat =    {:11.3f}   lon =    {:11.3f}   tmzone = {:11.1f}\n'.format(tuv_vo.lat, tuv_vo.lon, tuv_vo.tmzone)
    settings[4] = 'iyear =  {:11d}   imonth = {:11d}   iday =   {:11d}\n'.format(tuv_vo.sim_date.year, tuv_vo.sim_date.month, tuv_vo.sim_date.day)

    settings[7] = 'tstart = {:11.3f}   tstop =  {:11.3f}   nt =     {:11d}\n'.format(tuv_vo.tstart, tuv_vo.tstop, tuv_vo.nt)

    lzenit = 'F'
    if hasattr(tuv_vo, 'lzenit') and tuv_vo.lzenit:
        lzenit = 'T'
    settings[8] = 'lzenit =           {:1s}   alsurf = {:11.3f}   psurf =  {:11.1f}\n'.format(lzenit, tuv_vo.alsurf, tuv_vo.psurf)

    settings[9] = 'o3col =  {:11.3f}   so2col = {:11.3f}   no2col = {:11.3f}\n'.format(tuv_vo.o3col, tuv_vo.so2col, tuv_vo.no2col)

    settings[12-1] = 'tauaer = {:11.3f}   ssaaer = {:11.3f}   alpha =  {:11.3f}\n'.format(tuv_vo.tauaer, tuv_vo.ssaaer, tuv_vo.alpha)

    settings[13-1] = 'dirsun =       {:1.3f}   difdn =        {:1.3f}   difup =        {:1.3f}\n'.format(tuv_vo.dirsun, tuv_vo.difdn, tuv_vo.difup)

    laflux = 'F'
    lirrad='F'
    lmmech='F'
    if hasattr(tuv_vo, 'laflux') and tuv_vo.laflux:
        laflux = 'T'
    settings[15-1] = 'lirrad =           {:1s}   laflux =           {:1s}   lmmech =           {:1s}\n'.format(lirrad, laflux, lmmech)

    ljvals = 'F'
    if hasattr(tuv_vo, 'ljvals') and tuv_vo.ljvals:
        ljvals = 'T'
    settings[16] = 'ljvals =           {:1s}   ijfix =  {:11d}   nmj =    {:11d}\n'.format(ljvals, tuv_vo.ijfix, tuv_vo.nmj)

    # write settings out into a new file
    with open(tuv_input_fp, 'w') as file:
        file.writelines(settings)


def run_tuv(tuv_run_dir, suppress_output=False):
    tuv_exe = tuv_run_dir + 'tuv'
    tuv_args = ''

    stdout = None
    if suppress_output:
        stdout = subprocess.DEVNULL  # this will suppress the output
    process = subprocess.Popen(tuv_exe, cwd=tuv_run_dir, stdout=stdout)
    # process.wait()  # this will block execution until finished
    return process


def get_solar_noon(lat, lon, sim_day):
    """
    this function computes the solar noon for a given place and date and returns the zenith angle
    :param lat:
    :param lon:
    :return:
    """
    obs = ephem.Observer()
    obs.lat, obs.long = '{}'.format(lat), '{}'.format(lon)
    sun = ephem.Sun()

    sunrise = None
    noon = None
    sunset = None

    try:
        sunrise = obs.previous_rising(sun, start=sim_day).datetime()
        noon = obs.next_transit(sun, start=sunrise).datetime()
        sunset = obs.next_setting(sun, start=noon).datetime()
    except (ephem.NeverUpError, ephem.AlwaysUpError) as e:
        # this can happen at poles,  this mean that zenith angle will be negative at noon
        noon = obs.next_transit(sun, start=sim_day).datetime()

    obs.date = noon
    sun.compute(obs)
    # sun.alt  # zenith angle during solar noon

    return sun.alt, sunrise, noon, sunset


def get_UTC_offset(tuv_vo):
    utc_offset = None

    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=tuv_vo.lon, lat=tuv_vo.lat)
    # timezone_str = tf.timezone_at(lng=-123, lat=49)
    if timezone_str is None:
        utc_offset = 0
        print("Could not determine the time zone, setting 0")
    else:
        # Display the current time in that time zone
        timezone = pytz.timezone(timezone_str)
        utc_offset = timezone.utcoffset(tuv_vo.sim_date).total_seconds() / 60 / 60

    # UTC time zones vary from UTC-12 to UTC+14
    # TUV says time zone should be [-12;12]
    # TODO: don't have to fix it precisely now
    if utc_offset > 12:
        utc_offset -= 24
    return utc_offset


