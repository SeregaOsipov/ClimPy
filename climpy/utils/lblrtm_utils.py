import subprocess
import os
from enum import Enum
import struct
import numpy as np
import xarray as xr
import os.path
from climpy.utils.optics_utils import normalize_phase_function


class Gas(Enum):
    H2O = 'H2O'
    CO2 = 'CO2'
    O3 = 'O3'
    N2O = 'N2O'
    CO = 'CO'
    CH4 = 'CH4'
    O2 = 'O2'
    NO = 'NO'
    SO2 = 'SO2'
    NO2 = 'NO2'


# this gases comes in order from Table 2 of the rrtm_sw_instructions file, it is NOT a full list, only first part of it
AER_SUPPORTED_GASES = [Gas.H2O, Gas.CO2, Gas.O3, Gas.N2O, Gas.CO, Gas.CH4, Gas.O2, Gas.NO, Gas.SO2, Gas.NO2]


class LblrtmSetup(object):  # TODO: Temp dummy settings
    def __init__(self, DVOUT=10):
        self.DVOUT = DVOUT  # cm^-1
        self.zenithAngle = 180  # 0


def write_settings_to_tape(tape_fp, lblrtm_setup_vo, atm_stag_ds, gases_ds, cross_sections, include_Rayleigh_extinction):
    '''
    I think that LBLRTM takes input on STAGGERED grid and produces output on RHO grid.
    Since MERRA2 is given a fixed pressure levels, I treat them as STAGGERED grid.

    :param tape_fp: output file path
    :param atm_stag_ds:
    :param gases_ds: GMI output climatology  as DataArray
    :param include_Rayleigh_extinction:
    :return:
    '''

    IHIRAC = 1  # 1 - Voigt profile
    ILBLF4 = 0  # 1

    ICNTNM = 5  # all continua calculated, except Rayleigh extinction
    if include_Rayleigh_extinction:
        ICNTNM = 1

    IAERSL = 0
    IEMIT = 0  # optical     depth    only
    ISCAN = 0
    IFILTR = 0
    IPLOT = 0
    ITEST = 0
    IATM = 1
    IMRG = 1
    ILAS = 0
    IOD = 1  # 0
    IXSECT = 0
    if len(cross_sections.species) > 0:
        IXSECT = 1
    MPTS = 0
    NPTS = 0

    # IHIRAC, ILBLF4, ICNTNM, IAERSL,  IEMIT,  ISCAN, IFILTR, IPLOT, ITEST,  IATM,  IMRG,  ILAS,   IOD, IXSECT,  MPTS,  NPTS
    # 5,     10,     15,     20,     25,     30,     35,    40,    45,    50, 54-55,    60,    65,     70, 72-75, 77-80
    # 4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1, 4X,I1, 4X,I1, 4X,I1, 3X,A2, 4X,I1, 4X,I1,  4X,I1, 1X,I4, 1X,I4

    #%% output section
    tape = open(tape_fp, 'w+')
    tape.write('%c%s\n' % ('$', 'SW Era Interim, GMI, MODIS'))
    tape.write('%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d\n' % (IHIRAC, ILBLF4, ICNTNM, IAERSL, IEMIT, ISCAN,
            IFILTR, IPLOT, ITEST, IATM, IMRG, ILAS, IOD, IXSECT, MPTS, NPTS, MPTS, NPTS))

    if IHIRAC + IAERSL + IEMIT + IATM + ILAS > 0:
        V1 = lblrtm_setup_vo.V1  # remember V2 has to be bigger than V1
        V2 = lblrtm_setup_vo.V2
        if V2 - V1 > 2020:
            raise Exception('LBLRTM output: spectral range is wider than 2020 cm^-1')
        if V2 < V1:
            raise Exception('LBLRTM output: V2<V1, you probably dont want it. V2 is {} and V1 is {}'.format(V2, V1))

        SAMPLE = 4
        DVSET = 0  # 0.000007
        ALFAL0 = 0.04
        AVMASS = 36
        DPTMIN = 0.0002  # 0
        DPTFAC = 0.001  # 0
        ILNFLG = 0
        DVOUT = lblrtm_setup_vo.DVOUT
        print(DVOUT)
        NMOL_SCAL = 0

        # V1,     V2,   SAMPLE,   DVSET,  ALFAL0,   AVMASS,   DPTMIN,   DPTFAC,   ILNFLG,     DVOUT,   NMOL_SCAL
        # 1-10,  11-20,    21-30,   31-40,   41-50,    51-60,    61-70,    71-80,     85,      90-100,         105
        # E10.3,  E10.3,    E10.3,   E10.3,   E10.3,    E10.3,    E10.3,    E10.3,    4X,I1,  5X,E10.3,       3x,I2
        tape.write('%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%5d%5s%10.3E%5d\n' % (V1, V2, SAMPLE, DVSET, ALFAL0, AVMASS, DPTMIN, DPTFAC, ILNFLG, ' ', DVOUT, NMOL_SCAL))

    if IEMIT == 1:
        raise Exception('LBLRTM output: IEMIT=1 is not implemented and should not be')

        TBOUND = 300
        SREMIS = []
        SREMIS[1-1] = 0
        SREMIS[2-1] = 0
        SREMIS[3-1] = 0

        SRREFL = []
        SRREFL[1-1] = 0
        SRREFL[2-1] = 0
        SRREFL[3-1] = 0

        surf_refl = 'l'

        # TBOUND, SREMIS(1), SREMIS(2), SREMIS(3), SRREFL(1), SRREFL(2), SRREFL(3), surf_refl
        # 1-10,     11-20,     21-30,     31-40,     41-50,     51-60,     61-70,    75
        # E10.3,     E10.3,     E10.3,     E10.3,     E10.3,     E10.3,     E10.3    4X,1A
        # looks like I have to replace it with separate file with spectral properties
        tape.write('%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%5s\n', TBOUND, SREMIS(1), SREMIS(2), SREMIS(3), SRREFL(1), SRREFL(2), SRREFL(3), surf_refl);

    MODEL = 0
    ITYPE = 2

    IBMAX = -1 * len(atm_stag_ds.p)  # stag grid. TOTAL NUMBER OF LAYER boundaries
    print('{LBLRTMParser:outputSettings} Running in the z profile grid instead of p')
    IBMAX = len(atm_stag_ds.p)

    ZERO = 0  # option 2  zeroes absorber amounts which are less than 0.1 percent of total
    NOPRNT = 0
    NMOL = len(gases_ds.species)  # NUMBER OF    MOLECULAR    SPECIES
    IPUNCH = 1  # TAPE7 FILE OF MOLECULAR COLUMN AMOUNTS FROM LBLATM ONLY FOR IATM=1; IPUNCH=1 (CARD 3.1)
    IFXTYP = 0
    MUNITS = 0
    RE = 6371.23  # km
    HSPACE = 0  # atm_ds.z[-1] / 10 ** 3  # altitude definition for space
    VBAR = 0
    REF_LAT = atm_stag_ds.lat  # TODO: add check, has to have a single coordinate

    # MODEL, ITYPE, IBMAX, ZERO, NOPRNT, NMOL, IPUNCH, IFXTYP, MUNITS, RE, HSPACE, VBAR, REF_LAT
    # 5, 10, 15, 20, 25, 30, 35, 36 - 37, 39 - 40, 41 - 50, 51 - 60, 61 - 70, 81 - 90
    # I5,     I5,    I5,      I5,      I5,    I5,     I5,     I2,   1X, I2, F10.3,  F10.3, F10.3,   10x, F10.3
    tape.write('%5d%5d%5d%5d%5d%5d%5d%2d%3d%10.3f%10.3f%10.3f%10s%10.3f\n' % (MODEL, ITYPE, IBMAX, ZERO, NOPRNT, NMOL, IPUNCH, IFXTYP, MUNITS, RE, HSPACE, VBAR, ' ', REF_LAT))

    # for ITYPE = 2, only 3 of the first 5 parameters are required to specify the path, e.g., H1, H2, ANGLE or H1, H2 and RANGE
    # for ITYPE = 3, H1 = observer altitude must be specified.Either H2 = tangent height or ANGLE must be specified.Other parameters are ignored.
    # just test to see which one is higher, H1 or H2
    H1 = atm_stag_ds.z[-1] / 10 ** 3  # observer altitude
    H2 = atm_stag_ds.z[0] / 10 ** 3  # for ITYPE = 2, H2 is the end point altitude
    #H2 = atm_ds.z_sfc  # try surface height instead, since z_sfc is not included in the profile
    if IBMAX < 0:
        H1 = atm_stag_ds.p[-1]
        H2 = atm_stag_ds.p[0]
        #H2 = atm_ds.sp  # again, use surface instead of profile

    ANGLE = lblrtm_setup_vo.zenithAngle  # zenith angle at H1 (degrees)
    # ANGLE = swInputVO.zenithAngle + 90; % zenith
    RANGE = 0  # H2-H1; %length of a straight path from H1 to H2 (km)
    BETA = 0  # earth centered angle from H1 to H2 (degrees)
    LEN = 0
    HOBS = 0  # Height of observer, used only for informational purposes in satellite-type simulations when computing output geometry above 120 km.

    # H1, H2, ANGLE, RANGE, BETA, LEN, HOBS
    # 1 - 10, 11 - 20, 21 - 30, 31 - 40, 41 - 50, 51 - 55, 61 - 70
    # F10.3, F10.3,   F10.3,   F10.3,  F10.3,    I5, 5X,F10.3
    tape.write('%10.3f%10.3f%10.3f%10.3f%10.3f%5d%5s%10.3f\n' % (H1, H2, ANGLE, RANGE, BETA, LEN, ' ', HOBS))

    for j in range(abs(IBMAX)):
        format = '%10.3f'
        if j % 8 == 7 or j == abs(IBMAX)-1:
            format += '\n'
        if IBMAX > 0:
            tape.write(format % (atm_stag_ds.z[j] / 10 ** 3))  # altitudes of LBLRTM layer boundaries
        else:
            tape.write(format % atm_stag_ds.p[j])
    # tape.write('\n')

    IMMAX = len(atm_stag_ds.p)  # number of atmospheric profile boundaries
    IMMAX = IBMAX
    HMOD = 'profile description'
    # IMMAX, HMOD
    # 5, 6 - 29
    # I5, 3A8
    tape.write('%5d%s\n' % (IMMAX, ' '))

    write_gases_to_tape(atm_stag_ds, gases_ds, tape, IBMAX)  # TODO: previous implementation is missing JLONG variable

    if IXSECT > 0:  # write cross-sections
        IXMOLS = cross_sections.species.size
        IPRFL = 0
        IXSBIN = 0
        tape.write('%5d%5d%5d\n' % (IXMOLS, IPRFL, IXSBIN))
        for specie_index, specie in enumerate(cross_sections.species):
            gas_ds = cross_sections.sel(species=specie)
            format = '%10s'
            if specie_index % 8 == 7:
                format += '\n'
            tape.write(format % specie.item())
        tape.write('\n')

        LAYX = gas_ds.level.size
        IZORP = 0  # 0 - km, 1 - hPa
        # Sync cross-sections output with the user atmosphjere
        if IBMAX < 0:  # then atmospheric profile is in p.
            IZORP = 1
        XTITLE = 'major UV absorpers'
        tape.write('%5d%5d%50s\n' % (LAYX, IZORP, XTITLE))

        # assume that all the species has the same vertical grid
        for layer_index in range(LAYX):
            if IZORP==0:  # z, km
                tape.write('%10.3f%5s' % (atm_stag_ds.z[layer_index], ''))
            else:  # pressure, hPa
                tape.write('%10.3f%5s' % (gas_ds.level[layer_index], ''))
            # output units first
            for specie in cross_sections.species:
                gas_ds = cross_sections.sel(species=specie)
                tape.write('%1s' % 'A')
                if gas_ds.units.item() != 'A':
                    raise Exception('LBLRTM crosssections: LBL only accepts 1 or A units here')
            tape.write('\n')
            # now output densities
            for specie_index, specie in enumerate(cross_sections.species):
                gas_ds = cross_sections.sel(species=specie)
                format = '%10.3E'
                if specie_index % 8 == 7:
                    format = '+\n'
                tape.write(format % gas_ds.const[layer_index])
            tape.write('\n')

    tape.write('%%')
    tape.close()


def write_gases_to_tape(atm_ds, gases_ds, tape, ibmax):
    '''

    :param atm_ds:
    :param gases_ds:
    :param tape: is a file handle
    :param ibmax:
    :return:
    '''

    if (atm_ds.t<0).any() or (atm_ds.p<0).any() or (gases_ds.const<0).any():
        raise Exception('LBL: unphysical values (t, p or const < 0')

    # preprocessing (possibly could be done using xarray's sortby)
    ordered_gases = []  # sort gases in order required by model
    for specie_enum in AER_SUPPORTED_GASES:
        gas_ds = gases_ds.sel(species=specie_enum.value)  # missing gas will raise an exception
        # raise Exception('RRTMParser outputGasProperties: reguired gas {} is missing'.format(RRTMAbstractParser.rrtmSupportedGasesEnums[i].name))
        ordered_gases += [gas_ds, ]

    # ZM,    PM,    TM,    JCHARP, JCHART,   JLONG,   (JCHAR(M),M =1,39)
    # 1-10, 11-20, 21-30,        36,     37,      39,     41  through  80
    # E10.3, E10.3, E10.3,   5x,  A1,     A1,  1x, A1,     1x,    39A1

    for k in range(abs(ibmax)):
        tape.write('%10.3E' % (atm_ds.z[k] / 10**3,))
        tape.write('%10.3E' % atm_ds.p[k])
        tape.write('%10.3E' % atm_ds.t[k])

        tape.write('%6c%c' % ('A', 'A'))
        tape.write('%2c' % 'L')
        tape.write('%c' % ' ')

        for gas_ds in ordered_gases:
            tape.write('%c' % gas_ds.units.item())

        # tape.write('\n');
        for i, gas_ds in enumerate(ordered_gases):
            if i % 8 == 0:  # add \n every 8 items
                tape.write('\n')
            tape.write('%15.8E' % gas_ds.const[k])
        tape.write('\n')


def run_lblrtm(lblrtm_scratch_fp, lblrtm_setup_vo, atm_stag_ds, gases_ds, cross_sections, include_Rayleigh_extinction):
    '''
    Desc
    :param lblrtm_scratch_fp:
    :param lblrtm_setup_vo:
    :param atm_stag_ds:
    :param gases_ds:
    :param cross_sections:
    :param include_Rayleigh_extinction:
    :return:

    Run example:

    lblrtm_setup_vo = LblrtmSetup()
    lblrtm_setup_vo.V1 = 2000
    lblrtm_setup_vo.V2 = 4000
    lblrtm_setup_vo.DVOUT = 10
    lblrtm_setup_vo.zenithAngle = 0

    tape5_fp = '{}{}'.format(lblrtm_scratch_fp, 'TAPE5')
    # write_settings_to_tape(tape5_fp, lblrtm_setup_vo, atm_ds, gases_ds, cross_sections, True)
    run_lblrtm(lblrtm_scratch_fp, lblrtm_setup_vo, atm_ds, gases_ds, cross_sections, True)

    spectral_od_profile, wavelengts, wavenumbers = read_od_output(lblrtm_scratch_fp, atm_ds.level.size)
    '''

    # TODO: always remove previous files before rerunning
    print('TODO: always remove previous files before reruning !!!')
    #'\rm ODdef* TAPE3? TAPE6? TAPE?? TAPE7 TAPE9 TAPE5'
    #'\rm fort.601 fort.602 fort.603'

    # output TAPE5
    tape5_fp = '{}{}'.format(lblrtm_scratch_fp, 'TAPE5')
    write_settings_to_tape(tape5_fp, lblrtm_setup_vo, atm_stag_ds, gases_ds, cross_sections, include_Rayleigh_extinction)

    # local implementation
    # lblExecutablePath = '/work/mm0062/b302074/workspace/fortran/AER-RC/LBLRTM/lblrtm_v12.11_linux_intel_dbl'
    # lblExecutablePath = '/Users/osipovs/Temp/lblrtm_v12.11_linux_intel_dbl'  # TODO: this is just a copy from levante. Recompile locally
    # postfixString = ''
    # subprocess.run([lblExecutablePath, postfixString], cwd=lblrtm_scratch_fp)

    # remote implementation. Important to quote the command
    lblExecutableCmd = 'cd {} ; rm ./TAPE6 ODint_* ; ./lblrtm_v12.11_linux_intel_dbl'.format(lblrtm_scratch_fp)  # to execute lblrtm remotely
    result = subprocess.run('ssh levante "{}"'.format(lblExecutableCmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result)


def run_lblrtm_over_spectral_range(wn_range, lblrtm_scratch_fp, atm_stag_ds, gases_ds, cross_sections, lblrtm_setup_vo, include_Rayleigh_extinction=False):
    '''
    Remember that LBLRTM represent rho layer and optical properties of these layers.
    Work with wavenumbers instead of WLs. Wavenumbers in cm^-1. Wavelengths in um.

    # atm_ds (MERRA2) is on the staggered grid. Derived the profile on the rho grid
    atm_rho_ds = atm_stag_ds.rolling(level=2).mean().dropna('level')

    :param wn_range: is a list of [min_wn, max_wn]
    :param lblrtm_scratch_fp:
    :param lblrtm_setup_vo:
    :param atm_stag_ds:
    :param gases_ds:
    :param cross_sections:
    :param include_Rayleigh_extinction:
    :return:
    '''
    min_wn, max_wn = wn_range
    wn_step = 2000  # 1010  # max wn width is 2000 in LBLRTM
    wn_grid = np.arange(min_wn, max_wn, wn_step)
    if wn_grid[-1] != max_wn:  # last element is included in numpy arrange
        wn_grid = np.concatenate([wn_grid, [max_wn]])
        # I need to have a reasonalbe min distance between to points
        dwn = wn_grid[-1]-wn_grid[-2]
        if len(wn_grid) > 2 and dwn < 25:
            print('Last wavelength grid point is too close, adjusting. Original wn grid is {}'.format(wn_grid))
            wn_grid[-2] -= 25  # should be OK given the 2000 step
            print('New grid is {}'.format(wn_grid))
        # else: just two point should run OK

    print('Running LBLRTM over spectral range')
    ods = []
    wls = []
    wns = []
    for index in range(len(wn_grid)-1):
        lblrtm_setup_vo.V1 = wn_grid[index]
        lblrtm_setup_vo.V2 = wn_grid[index+1]

        print('Spectral interval {}/{}: {} to {} cm^-1'.format(index+1, len(wn_grid)-1, lblrtm_setup_vo.V1, lblrtm_setup_vo.V2))

        # input is on the staggered grid
        run_lblrtm(lblrtm_scratch_fp, lblrtm_setup_vo, atm_stag_ds, gases_ds, cross_sections, include_Rayleigh_extinction)
        # output is on the rho grid. Thus, n levels is = n stag levels - 1
        spectral_od_profile, wavelengts, wavenumbers = read_od_output(lblrtm_scratch_fp, atm_stag_ds.level.size-1)

        ods += [spectral_od_profile, ]
        wls += [wavelengts, ]
        wns += [wavenumbers, ]

    spectral_od_profile = np.concatenate(ods, axis=1)
    wavelengts = np.concatenate(wls, axis=0)
    wavenumbers = np.concatenate(wns, axis=0)

    # convert into full set of optical properties of gas mixture
    od = spectral_od_profile
    ssa = np.zeros(od.shape)
    g = np.zeros(od.shape)
    # this is absorption only of the trace gases and does not have a phase function. I'm gonna setup uniform ones here for numerical reasons
    phase_function_angles = np.linspace(0, np.pi, 180)
    phase_function = np.ones(od.shape + phase_function_angles.shape)
    # normalize
    mu = np.cos(phase_function_angles)
    norm = 2 * np.pi * np.trapz(np.ones(phase_function_angles.shape), mu)
    norm *= -1/2
    phase_function /= norm

    # prep the levels at RHO grid since # LBLRTM output is RHO grid, while INPUT is STAGGERED
    levels_rho = atm_stag_ds.level.rolling(level=2).mean().dropna('level')

    ds = xr.Dataset(
        data_vars=dict(
            od=(["level", "wavenumber"], od),
            ssa=(["level", "wavenumber"], ssa),
            g=(["level", "wavenumber"], g),
            phase_function=(["level", "wavenumber", "angle"], phase_function),
        ),
        coords=dict(
            level=(['level', ], levels_rho.data),
            wavelength=(['wavenumber', ], wavelengts),
            wavenumber=(['wavenumber', ], wavenumbers),
            angle=(['angle', ], phase_function_angles),
        ),
        attrs=dict(description="Optical properties according to LBLRTM"),
    )

    # the intervaling produces duplicate values in WL. Get rid of them
    unique_wns, ind = np.unique(wavenumbers, return_index=True)
    ds = ds.isel(wavenumber=ind)

    return ds


def read_tape11_output(tape_fp, opt):
    '''
    % File format illustration
    % for single precision
    % shift 266*4 bytes
    % LOOP
    % 1 int        , 24 (block of v1, v2, dv, npts)
    % 2 double vars, for v1, and v2
    % 1 float      , for dv
    % 1 int        , for npts
    % 1 int        , 24
    % 1 int        , 9600 or npts*4 (beg of block output)
    % NPTs float   , rad
    % 1 int        , 9600 or npts*4 (end of block of output)
    % LOOP ENDS

    % for double precision
    % shift 356*4 bytes
    % LOOP
    % 1 int        , 32 (v1, v2, dv and npts, extra 0)
    % 3 double vars, for v1, v2, and dv
    % 1 long int   , for npts
    % 1 int        , 32
    % 1 int        , 19200 or npts*8 (beg of block of output)
    % NPTS double  , rad
    % 1 int        , 19200 or npts*8 (end of block of output)
    % LOOP ENDS

    % Author: Xianglei Huang
    % Tested on Redhat Linux with pgi-compiler version of LBLRTM
    % ported by Sergey Osipov from Matlab to Python
    '''

    v = []
    rad = []

    fid = open(tape_fp, mode='rb')

    if opt[0].lower() == 'f' or opt[0].lower() == 's':
        shift = 266
        itype = 1
    else:
        shift = 356
        itype = 2

    fid.seek(shift*4)
    test = struct.unpack('i', fid.read(4))

    little_endian = False  # in most cases it is a little endian.
    if (itype == 1 and test == 24) or (itype == 2 and test == 32):
        little_endian = True
    # else big endian

    endflg = 0
    panel = 0

    if itype == 1:
        while endflg == 0:
            raise('Not ported completely')
            panel += 1
            # disp(['read panel ', int2str(panel)])
            v1, = struct.unpack('d', fid.read(8))  # 1, 'double');
            if isnan(v1):
                break
            v2, = struct.unpack('d', fid.read(8))  # 1, 'double');
            dv, = struct.unpack('f', fid.read(4))  # 1, 'float');
            npts, = struct.unpack('i', fid.read(4))  # 1, 'int');
            fid.read(4)

            LEN, = struct.unpack('i', fid.read(4))
            if LEN != 4 * npts:
                raise('internal file inconsistency')
                endflg = 1
            tmp, = struct.unpack('f'*npts, fid.read(4*npts))  # , npts, 'float');
            LEN2, = struct.unpack('i', fid.read(4))
            if LEN != LEN2:
                raise('internal file inconsistency')
                endflg = 1
            v += [v1, v2, dv]  # this concatenation is probably in wrong dimensions, check
            rad += tmp
    else:
        while endflg == 0:
            panel += 1
            # disp(['read panel ', int2str(panel)]);
            v1, v2, dv = struct.unpack('ddd', fid.read(8*3))  # 3, 'double');
            if np.isnan(v1):
                break
            npts, = struct.unpack('q', fid.read(8))  # 1, 'int64');  # q or Q

            if npts != 2400:
                endflg = 1

            struct.unpack('i', fid.read(4))  # 1, 'int')
            LEN, = struct.unpack('i', fid.read(4))  # 1, 'int')
            if LEN != 8 * npts:
                raise('internal file inconsistency')  # or print
                endflg = 1

            tmp = struct.unpack('d'*npts, fid.read(8*npts))  # npts, 'double')
            LEN2, = struct.unpack('i', fid.read(4))
            if LEN != LEN2:
                raise('internal file inconsistency')
                endflg = 1

            v += [v1, v2, dv]  # this concatenation is probably in wrong dimensions, check
            rad += tmp
    fid.close()
    return np.array(v), np.array(rad)


def read_od_output(lblrtm_scratch_fp, n_layers_in_profile):

    spectral_items = []
    od_item_shape = None
    for layer_index in range(n_layers_in_profile):
        od_file_path = '{}/ODint_{:03d}'.format(lblrtm_scratch_fp, layer_index+1)
        tape6_file_path = '{}/TAPE6'.format(lblrtm_scratch_fp)

        spectral_item = None
        if os.path.exists(od_file_path):
            # first check that it does not have any warnings
            with open(tape6_file_path) as f:
                if 'WARNING' in f.read():
                    raise Exception('LBLRTM containts WARNINGS in TAPE6. Deal with them first.')

            v, spectral_item = read_tape11_output(od_file_path, 'double')
            v1 = v[0]
            v2 = v[1]
            dv = v[2]
            od_item_shape = spectral_item.shape  # note the array shape for missing layers
        else:  # it is possible that layer does not exist because it was zeroed out by LBLRTM
            print('LBLRTM output for layer {} is missing in file path {}\nAssuming zeros'.format(layer_index+1, od_file_path))
            if od_item_shape is None:
                raise Exception('LBLRTM output shape is unknown, likely something is wrong')  # This means that first layer for zeroes out, which should not happen
            spectral_item = np.zeros(od_item_shape)
        spectral_items += [spectral_item,]

    # precaution: make sure that we read all layers. There should not be anymore ODint_ files left
    od_file_path = '{}/ODint_{:03d}'.format(lblrtm_scratch_fp, layer_index + 1 + 1)
    if os.path.exists(od_file_path):
        raise Exception('lblrtm_utils:read_od_output. More LBLRTM output files (layers) than there should be. Check what is wrong.')

    spectral_od_profile = np.array(spectral_items)
    wavenumbers = np.linspace(v1, v2, spectral_item.shape[0])
    wavelengts = 10** 4. / wavenumbers

    return spectral_od_profile, wavelengts, wavenumbers