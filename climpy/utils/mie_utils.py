import miepython as mp
import numpy as np
import xarray as xr
import time
__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def get_mie_efficiencies_numpy(ri, r_data, wavelength, phase_function_angles_in_radians=None):
    """
    ri = ri_vo['ri'][ind]
    r_data = dA_vo['radii']
    wavelength = ri_vo['wl']

    :param ri:
    :param r_data: units should be the same as wavelength
    :param wavelength: units should be the same as radius
    :return:  mie efficiecnies units are microns^2, assuming the input in microns
    """

    # mp.mie can take x vector as input
    # these are the ext and sca efficiencies, compare to cross sections (qext * A)
    qext = np.empty((len(wavelength), len(r_data)))
    qsca = np.empty((len(wavelength), len(r_data)))
    g = np.empty(qext.shape)
    qasm = np.empty(qext.shape)

    if phase_function_angles_in_radians is None:
        phase_function_angles_in_radians = np.linspace(0, np.pi, 180)  # angles to evaluate the phase function
    phase_function = np.empty(qext.shape + phase_function_angles_in_radians.shape)  # wl, r, angle

    for wl_index in range(len(wavelength)):
        # print('Wavelength {} out of {}'.format(wl_index, len(wavelength)))
        # t_s = time.time()
        for r_index in range(len(r_data)):
            m = ri[wl_index]
            # miepython sign convention: imaginary part has to be negative; complex RI = real -j * imaginary
            m = m.real - 1j * np.abs(m.imag)
            x = 2 * np.pi * r_data[r_index] / wavelength[wl_index]
            # efficiencies are without units, area will be in um^2
            qext[wl_index, r_index], qsca[wl_index, r_index], qback, g[wl_index, r_index] = mp.mie(m.item(),x.item())

            # compute the phase function
            phase_function[wl_index, r_index, :] = mp.i_unpolarized(m.item(), x.item(), np.cos(phase_function_angles_in_radians))  # this is the phase function normalized to ssa
        # t_e = time.time()
        # print(t_e-t_s)

    if np.any(qext < 0):
        raise Exception('Mie solution can not have negative extinction values, check RI')

    return qext, qsca, g, qasm, phase_function, phase_function_angles_in_radians  # , always specify angles to avoid issues with xarray aply_ufunc


def get_mie_efficiencies(ri, r_data, wavelength, phase_function_angles_in_radians=None):
    qext, qsca, g, qasm, phase_function, phase_function_angles_in_radians = get_mie_efficiencies_numpy(ri, r_data, wavelength, phase_function_angles_in_radians)

    mie_ds = xr.Dataset(
        data_vars=dict(
            qext=(['wavenumber', 'radius'], qext),
            qsca=(['wavenumber', 'radius'], qsca),
            g=(["wavenumber", 'radius'], g),
            phase_function=(["wavenumber", 'radius', "angle"], phase_function),
        ),
        coords=dict(
            radius=(['radius', ], r_data.data),
            wavelength=(['wavenumber', ], wavelength.data),
            wavenumber=(['wavenumber', ], 10**4/wavelength.data),
            angle=(['angle', ], phase_function_angles_in_radians),
        ),
        attrs=dict(description="Optical properties for a given size distribution"),
    )

    return mie_ds


def integrate_in_log_r(ds):
    # temp_ds = ds.copy(deep=True)
    # temp_ds['radius'] = np.log(temp_ds.radius)
    # result = temp_ds.integrate(coord='radius')

    # alternative approach
    ds = ds.assign_coords(log_radius=('radius', np.log(ds['radius'].data)))
    result = ds.integrate(coord='log_radius')

    return result


def integrate_mie_over_aerosol_size_distribution(mie_ds, dN_ds, include_phase_function=True):
    '''
    Assumptions about units:
    The input (radius, wavelength) in microns
    Mie efficiencies are dimensionless and cross-sections are microns^2

    dNdlnr has units of [number / um^2] (column) or [number / um^3] (profile)

    :param mie_ds:
    :param dN_ds:
    :param include_phase_function: has large memory footprint
    :return:
    '''

    dNdlnr = dN_ds.dNdlogd  # [number / um^2] (column) or [number / um^3] (profile)
    A = np.pi * mie_ds.radius ** 2  # um^2

    # Have to integrate in log(r), not just r
    ext_coefficient = integrate_in_log_r(dNdlnr * A * mie_ds.qext)
    sca_coefficient = integrate_in_log_r(dNdlnr * A * mie_ds.qsca)
    asy_coefficient = integrate_in_log_r(dNdlnr * A * mie_ds.g * mie_ds.qsca)  # qasm = asymmetryParameter. * A. * qsca;
    if include_phase_function:
        phase_function = integrate_in_log_r(dNdlnr * A * mie_ds.phase_function * mie_ds.qsca) / sca_coefficient

    ssa = sca_coefficient / ext_coefficient
    g = asy_coefficient / sca_coefficient

    das = [ext_coefficient.rename('ext'), ssa.rename('ssa'), g.rename('g')]
    if include_phase_function:
        das.append(phase_function.rename('phase_function'))

    op_ds = xr.merge(das)

    return op_ds



