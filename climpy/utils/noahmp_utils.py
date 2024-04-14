import f90nml
import pandas as pd
import xarray as xr
import numpy as np

def get_noahmp_table_as_df(noahmp_table_fp = '/project/k1090/osipovs/Models/hrldas/noahmp/parameters/NoahmpTable.TBL'):
    '''

    Prepare the NoahMP parameters from the table

    Line 799 in /home/osipovs/PycharmProjects/hrldas/noahmp/drivers/hrldas/NoahmpReadTableMod.F90

    ! assign values
    NoahmpIO%SLCATS_TABLE           = SLCATS
    NoahmpIO%BEXP_TABLE  (1:SLCATS) = BB    (1:SLCATS)
    NoahmpIO%SMCDRY_TABLE(1:SLCATS) = DRYSMC(1:SLCATS)
    NoahmpIO%SMCMAX_TABLE(1:SLCATS) = MAXSMC(1:SLCATS)
    NoahmpIO%SMCREF_TABLE(1:SLCATS) = REFSMC(1:SLCATS)
    NoahmpIO%PSISAT_TABLE(1:SLCATS) = SATPSI(1:SLCATS)
    NoahmpIO%DKSAT_TABLE (1:SLCATS) = SATDK (1:SLCATS)
    NoahmpIO%DWSAT_TABLE (1:SLCATS) = SATDW (1:SLCATS)
    NoahmpIO%SMCWLT_TABLE(1:SLCATS) = WLTSMC(1:SLCATS)
    NoahmpIO%QUARTZ_TABLE(1:SLCATS) = QTZ   (1:SLCATS)
    NoahmpIO%BVIC_TABLE  (1:SLCATS) = BVIC  (1:SLCATS)
    NoahmpIO%AXAJ_TABLE  (1:SLCATS) = AXAJ  (1:SLCATS)
    NoahmpIO%BXAJ_TABLE  (1:SLCATS) = BXAJ  (1:SLCATS)
    NoahmpIO%XXAJ_TABLE  (1:SLCATS) = XXAJ  (1:SLCATS)
    NoahmpIO%BDVIC_TABLE (1:SLCATS) = BDVIC (1:SLCATS)
    NoahmpIO%GDVIC_TABLE (1:SLCATS) = GDVIC (1:SLCATS)
    NoahmpIO%BBVIC_TABLE (1:SLCATS) = BBVIC (1:SLCATS)

    :param noahmp_table_fp:
    :return:
    '''
    noahmp_soil_stas_parameters = f90nml.read(noahmp_table_fp)['noahmp_soil_stas_parameters']


    noahmp_table_df = pd.DataFrame(data={'BEXP':noahmp_soil_stas_parameters['bb'],
                                         'DKSAT':noahmp_soil_stas_parameters['SATDK'],
                                         'PSISAT':noahmp_soil_stas_parameters['SATPSI'],
                                         'SMCMAX':noahmp_soil_stas_parameters['MAXSMC'],
                                         'SMCWLT':noahmp_soil_stas_parameters['WLTSMC'],})

    noahmp_table_df.index.rename('ISLTYPE', inplace=True)

    return noahmp_table_df


def span_noahmp_parameters_onto_dataset(ds):
    '''
    Make sure:
     ds = ds.isel(time=0)

    :param ds:
    :return:
    '''
    noahmp_table_df = get_noahmp_table_as_df()
    noahmp_table_ds = noahmp_table_df.to_xarray()
    noahmp_table_ds = noahmp_table_ds.expand_dims(dim={'south_north':ds.south_north, 'west_east':ds.west_east}, axis=(1,2))  # here I'm assuming wrf input

    isltype = ds.ISLTYP
    isltype.load()
    isltype_index = isltype - 1  # remember that indexing is 0 based

    bexp = noahmp_table_ds.BEXP[isltype_index]
    psisat = noahmp_table_ds.PSISAT[isltype_index]
    smcmax = noahmp_table_ds.SMCMAX[isltype_index]
    smcwlt = noahmp_table_ds.SMCWLT[isltype_index]

    # zsoil = -1*ic_ds.DZS.cumsum()  # TODO: zsoil is missing 0 as first layer
    # # zsoil = xr.concat([0, zsoil], dim='soil_layers_stag
    # wtd = -1 * ds.ZWT
    #
    # wgpmid = smcmax * (psisat / (psisat - (zsoil[-1]-wtd)))**(1/bexp)
    # # wgpmid = np.maximum(wgpmid, smcwlt)
    # wgpmid = np.maximum(wgpmid, 1.E-4)
    #
    # syielddw = smcmax-wgpmid
    # totwater = wtd*syielddw

    noahmp_ds = xr.merge([bexp, psisat, smcmax, smcwlt])  #noahmp parameters expanded to input dataset dimensions

    return noahmp_ds


def derive_additional_noahmp_diags(ds, ic_ds):
    ds['soil_water_storage'] = 10**3 * (ds['SOIL_M'] * ic_ds.DZS).sum(dim='soil_layers_stag')  # 10**3 * m3/m3 * m = mm
    ds['soil_water_storage'] = ds['soil_water_storage'].assign_attrs(units='mm', description='Water storage in soil')

    ddz = -1*ic_ds.DZS.sum(dim='soil_layers_stag') - ds.ZWT  # zwt is negative, dzs is positive
    ds['deep_soil_water_storage'] = 10**3 * (ds['SMCWTD'] * ddz)  # 10**3 * m3/m3 * m = mm  # .sum(dim='soil_layers_stag')  # lets keep the vertical profile information
    ds['deep_soil_water_storage'] = ds['deep_soil_water_storage'].assign_attrs(units='mm', description='Water storage between soil and WT')

    # Next is auxiliary variable when only makes sense as chagne and not as absolute value
    # noahmp_ds = span_noahmp_parameters_onto_dataset(ds.isel(Time=0))
    # wgpmid = noahmp_ds.SMCMAX * ( noahmp_ds.PSISAT / (noahmp_ds.PSISAT - ddz) ) ** (1./noahmp_ds.BEXP)
    # wgpmid = np.maximum(wgpmid, 1.E-4)
    # noahmp_ds['WGPMID'] = wgpmid
    #
    # syielddw = noahmp_ds.SMCMAX-noahmp_ds.WGPMID
    # ds['water_storage_to_wt'] = 10**3 * ddz/syielddw  # 10**3 * m3/m3 * m = mm
    # ds['water_storage_to_wt'] = ds['water_storage_to_wt'].assign_attrs(units='mm', description='Water storage (potential) between surface and WT')

    # ds['water_storage_to_wt'] = 10**3 * noahmp_ds.SMCMAX * (-1*ds.ZWT) # 10**3 * m3/m3 * m = mm
    # ds['water_storage_to_wt'] = ds['water_storage_to_wt'].assign_attrs(units='mm', description='Water storage (potential) between surface and WT')

    ds['IRSMFIVOL'] = ds['IRSIVOL'] + ds['IRMIVOL'] + ds['IRFIVOL']
    ds['IRSMFIVOL'].attrs['description'] = 'Sprinkler + Micro + Flood irrigation amount'
    ds['IRSMFIVOL'].attrs['units'] = ds['IRSIVOL'].units

    return ds
