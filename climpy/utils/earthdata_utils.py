def check_and_generate_earthdata_netrc():
    '''
    See description in https://disc.gsfc.nasa.gov/information/howto?keywords=%22Earthdata%20Cloud%22&title=How%20to%20Directly%20Access%20MERRA-2%20Data%20from%20an%20S3%20Bucket%20with%20Python%20from%20a%20Cloud%20Environment
    :return:
    '''

    from netrc import netrc
    from subprocess import Popen
    import os
    from getpass import getpass

    urs = 'urs.earthdata.nasa.gov'    # Earthdata URL to call for authentication
    prompts = ['Enter NASA Earthdata Login Username \n(or create an account at urs.earthdata.nasa.gov): ',
               'Enter NASA Earthdata Login Password: ']

    # Determine if netrc file exists, and if so, if it includes NASA Earthdata Login Credentials
    try:
        netrcDir = os.path.expanduser("~/.netrc")
        netrc(netrcDir).authenticators(urs)[0]

    # Below, create a netrc file and prompt user for NASA Earthdata Login Username and Password
    except FileNotFoundError:
        homeDir = os.path.expanduser("~")
        Popen('touch {0}.netrc | chmod og-rw {0}.netrc | echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(getpass(prompt=prompts[0]), homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(getpass(prompt=prompts[1]), homeDir + os.sep), shell=True)

    # Determine OS and edit netrc file if it exists but is not set up for NASA Earthdata Login
    except TypeError:
        homeDir = os.path.expanduser("~")
        Popen('echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(getpass(prompt=prompts[0]), homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(getpass(prompt=prompts[1]), homeDir + os.sep), shell=True)


def get_credentials_from_netrc():  # get the password from ~/.netrc
    import os
    netrc_fp = os.path.expanduser("~/.netrc")
    with open(netrc_fp) as f:
        line = f.readline()
    parts = line.split(' ')
    username = parts[-3]
    password = parts[-1]
    # username = "username" # replace with your EarthData username
    # password = "password"  # replace with your EarthData password

    return username, password


def open_xarray_dataset_at_earthdata(url):
    from pydap.client import open_url
    from pydap.cas.urs import setup_session
    import xarray as xr

    username, password = get_credentials_from_netrc()
    session = setup_session(username, password, check_url=url)
    pydap_ds = open_url(url, session=session)
    store = xr.backends.PydapDataStore(pydap_ds)
    ds = xr.open_dataset(store)

    return ds