import xarray as xr
import numpy as np
'''
function rayleighOpticalPropertiesVO = getRayleighOpticalProperties(odData, waveLengthData, inputSettingsVO)    
            numberOfAngles = 100;
            pfSize = cat(2, size(rayleighOpticalPropertiesVO.odData), numberOfAngles);

            rayleighOpticalPropertiesVO.phaseFunctionAnglesData = linspace(0, pi, numberOfAngles);
            u=cos(rayleighOpticalPropertiesVO.phaseFunctionAnglesData);
            pfData = OpticsController.getRayleighPhaseFunction(u);
            pfData = reshape(pfData, [1 1 numberOfAngles]);
            pfData = repmat(pfData, [pfSize(1) pfSize(2) 1]);

            rayleighOpticalPropertiesVO.phaseFunctionData = pfData;
            rayleighOpticalPropertiesVO.layersPressureData = inputSettingsVO.layersPressures;      
            rayleighOpticalPropertiesVO.boundariesPressureData = inputSettingsVO.boundariesPressures;
        end  
'''


def derive_rayleigh_phase_function(angle_in_radians):
    '''
    The Rayleigh scattering phase function moments:
    # PMOM = single(zeros([NMOM+1, NLYR]));
    # PMOM(1,:) = 1;
    # PMOM(3,:) = 0.1;
    :return:
    '''

    pf = 3 / 4 * (1 + np.cos(angle_in_radians) ** 2)
    pf = normalize_phase_function(pf, angle_in_radians)

    return pf

def normalize_phase_function(pf, angle_in_radians):
    # TODO: redo using Xarray
    # normalize PF it to 1.  # https://miepython.readthedocs.io/en/latest/03a_normalization.html
    mu = np.cos(angle_in_radians)
    total = 2 * np.pi * np.trapz(pf, mu)
    if mu[1] < mu[0]:  # 0 to pi angle, 1 to -1 cos(angle)
        total *= -1  # Invert the sign due to reversed direction of integration.

    # TODO: I'm not sure what the normalization for DISORT should be (1, ssa, 2pi, etc.)
    total /= 2  # this is how norm was defined in the previous matlab code
    pf /= total  # normalize to 1
    return pf


def derive_rayleigh_optical_properties(op_ds_w_rayleigh, op_ds_wo_rayleigh):
    '''

    TODO: Would be better parametrize Rayleigh
    E.g., Rayleigh scattering is following the Nicolet 1984 https://doi.org/10.1016/0032-0633(84)90089-8
    https://doi.org/10.1175/1520-0426(1999)016%3C1854:ORODC%3E2.0.CO;2

    :param op_ds_w_rayleigh: is the LBLRTM with Rayleigh included
    :param op_ds_wo_rayleigh: the same, without Rayleigh scattering
    :return:
    '''
    rayleigh_od_da = op_ds_w_rayleigh.od - op_ds_wo_rayleigh.od
    # TODO: maybe it is better do make a deep copy and edit from there
    rayleigh_op_ds = get_rayleigh_optical_properties(rayleigh_od_da, op_ds_w_rayleigh.angle.data)  # op_ds store the angle in radiance

    return rayleigh_op_ds


def get_rayleigh_optical_properties(rayleigh_od_da, angle_in_radians):
    ssa = np.ones(rayleigh_od_da.shape)
    g = np.zeros(rayleigh_od_da.shape)  # alternative is to derive from PF
    phase_function = derive_rayleigh_phase_function(angle_in_radians)
    phase_function = np.tile(phase_function, rayleigh_od_da.shape[0:2] + (1,))

    op_ds_rayleigh = xr.Dataset(
        data_vars=dict(
            od=(["level", 'wavenumber'], rayleigh_od_da.data),
            ssa=(["level", 'wavenumber'], ssa),
            g=(["level", 'wavenumber'], g),
            phase_function=(["level", 'wavenumber', "angle"], phase_function),
        ),
        coords=dict(
            level=(['level', ], rayleigh_od_da.level.data),
            wavenumber=(['wavenumber', ], rayleigh_od_da.wavenumber.data),
            wavelength=(['wavenumber', ], rayleigh_od_da.wavelength.data),
            angle=(['angle', ], angle_in_radians),  # op_ds stores the angle in radiance
        ),
        attrs=dict(description="Rayleigh optical properties"),
    )

    return op_ds_rayleigh


def mix_optical_properties(ops, externally=True):
    '''
    Mix optical properties using the external mixture rules
    :param ops: list of op_ds
    :param externally:
    :return:
    '''

    bulk_ds = xr.concat(ops, dim='species')
    # aod is just a sum
    od_ds = bulk_ds.od.sum(dim='species')

    # ssa is weighted by extinction
    weights = bulk_ds.od/bulk_ds.od.sum(dim='species')
    weights.name = 'weight'
    if (bulk_ds.od.sum(dim='species')==0).any():  # then I will get NaN due to division by 0
        weights = weights.fillna(0)
    ssa_ds = bulk_ds.ssa.weighted(weights).sum(dim='species')

    # g and phase function are weighted by extinction * ssa
    od_times_ssa_ds = bulk_ds.od * bulk_ds.ssa
    weights = od_times_ssa_ds / od_times_ssa_ds.sum(dim='species')
    weights.name = 'weight'
    if (od_times_ssa_ds.sum(dim='species')==0).any():  # then I will get NaNs due to division by 0
        weights = weights.fillna(0)
    g_ds = bulk_ds.g.weighted(weights).sum(dim='species')
    pf_ds = bulk_ds.phase_function.weighted(weights).sum(dim='species')

    # put everything back together
    ds = xr.merge((od_ds, ssa_ds, g_ds, pf_ds))

    return ds

