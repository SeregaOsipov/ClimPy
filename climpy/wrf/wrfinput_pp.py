import xarray as xr

'''
Fix inconsitent data values in wrfinput
In this case, AREAD campaign, wrfinput has missing values in SNOW variable
'''

#%%
fp = '/work/mm0062/b302074/Data/AirQuality/AREAD/IC_BC/debugICBC/wrfinput_d01'
#%% fix the snow issues
df = xr.open_dataset(fp)  # consider using , autoclose=True
var_key = 'SNOW'
print('var {}: filling snow NaN values with 0'.format(var_key))
df[var_key].load()
df[var_key] = df[var_key].fillna(0)
# df[var_key][0,190,177]  # nan location
#%%
df.close()  # have to close to write update
df[var_key].to_netcdf(fp, mode='a')

#%%
df = xr.open_dataset(fp)  # consider using , autoclose=True
var_key = 'SNOWH'
print('var {}: filling snow NaN values with 0'.format(var_key))
df[var_key].load()
df[var_key] = df[var_key].fillna(0)
# df[var_key][0,190,177]  # nan location
#%%
df.close()  # have to close to write update
df[var_key].to_netcdf(fp, mode='a')
print('DONE')


#%% replace wrfinput date
fp1 = '/work/mm0062/b302074/AirQuality/AQABA/WRF/run_aread_100/wrfinput_d01'
df1 = xr.open_dataset(fp1)
fp2 = '/work/mm0062/b302074/AirQuality/AQABA/WRF/run_real/wrfinput_d01'
df2 = xr.open_dataset(fp2)

var_key = 'Times'
df1[var_key].load()
df1[var_key][:] = df2[var_key][:]
df1.close()
df1[var_key].to_netcdf(fp1, mode='a')
print('DONE')