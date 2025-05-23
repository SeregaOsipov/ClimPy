import xarray as xr

'''
The script provides python analog of ncrcat implemented via xarray
'''

def do_ncrcat(fp_in, fp_out):
    print('fp_ind: {}'.format(fp_in))
    print('fp_out: {}'.format(fp_out))

    ds = xr.open_mfdataset(fp_in)
    ds.to_netcdf(fp_out)


#%% Process Air Quality Projections, merge hourly into one file
# for scenario in '2050/HLT 2050/CLE 2017 2050/MFR 2050/MFR_LV'.split(' '):
for scenario in '2017 2050/MFR 2050/MFR_LV'.split(' '):
    print('scenario: {}'.format(scenario))

    fp_in = '/work/mm0062/b302074/Data/AirQuality/EMME/{}/chem_100_v1/output/pop_wtd/wrfout_d01_20*'.format(scenario)
    fp_out = '/work/mm0062/b302074/Data/AirQuality/EMME/{}/chem_100_v1/output/pop_wtd/wrfout_d01_hourly_pop_wtd'.format(scenario)
    do_ncrcat(fp_in, fp_out)
print('DONE')
#%%