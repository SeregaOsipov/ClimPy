import xarray as xr
import numpy as np
from climpy.merra2.merra2_aod_anth_natural_diags import derive_aod_fractions, get_sulfate_fractions, \
    get_sea_salt_fractions, get_dust_fractions, get_oc_fractions, get_bc_fractions
from datetime import datetime
import argparse


__author__ = 'Sergey Osipov <Sergey.Osipov@kaust.edu.sa>'

'''
Script will derive MERRA2 AOD diags for each date
Run example (single):
python ${CLIMPY}/climpy/merra2/merra2_aod_anth_natural_pp.py --date=19910615
Run example (batch):
Use ${CLIMPY}/climpy/merra2/run_merra2_pp_in_batch.sh to process in bulk
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
#parser.add_argument("--wrf_in", help="wrf input file path")   # , default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/wrfout_d01_2017-06-15_00:00:00')
parser.add_argument("--date", help="date to process YYYY-MM-DD", required=True)  # , default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/pp_wrf/wrfout_d01_2017-06-15_00:00:00')
args = parser.parse_args()

date_to_process = datetime.strptime(args.date, "%Y-%m-%d")

print('Process {} '.format(date_to_process.strftime('%Y-%m-%d')))

data_root = '/work/mm0062/b302074/Data/NASA/MERRA2/'
out_file_path = '{}/pp/aod_diags_{}.nc4'.format(data_root, date_to_process.strftime('%Y%m%d'))

#%%

sulfate_natural_frac, sulfate_anth_frac = get_sulfate_fractions(date_to_process)
sea_salt_natural_frac, sea_salt_anth_frac = get_sea_salt_fractions(date_to_process)
dust_natural_frac, dust_anth_frac = get_dust_fractions(date_to_process)
oc_natural_frac, oc_anth_frac = get_oc_fractions(date_to_process)
bc_natural_frac, bc_anth_frac = get_bc_fractions(date_to_process)

keys = ('DUEXTTAU', 'SSEXTTAU', 'SUEXTTAU', 'OCEXTTAU', 'BCEXTTAU')
anth_fracs = (dust_anth_frac, sea_salt_anth_frac, sulfate_anth_frac, oc_anth_frac, bc_anth_frac)
natural_fracs = (dust_natural_frac, sea_salt_natural_frac, sulfate_natural_frac, oc_natural_frac, bc_natural_frac)

anth_aod_frac, natural_aod_frac, anth_aod, natural_aod = derive_aod_fractions(date_to_process, keys, anth_fracs, natural_fracs)

# derive 1d diags too
cos_weight = np.cos(np.deg2rad(natural_aod['lat']))
cos_weight.name = 'weight'

total_aod = anth_aod + natural_aod
anth_aod_frac_1d = anth_aod.weighted(cos_weight).mean(dim=['lat', 'lon']) / total_aod.weighted(cos_weight).mean(dim=['lat', 'lon'])
natural_aod_frac_1d = 1-anth_aod_frac_1d
anth_aod_frac_1d = anth_aod_frac_1d.rename('anth_aod_frac_1d')
natural_aod_frac_1d = natural_aod_frac_1d.rename('natural_aod_frac_1d')

aod_1d = total_aod.weighted(cos_weight).mean(dim=['lat', 'lon'])
anth_aod_1d = anth_aod.weighted(cos_weight).mean(dim=['lat', 'lon'])
natural_aod_1d = natural_aod.weighted(cos_weight).mean(dim=['lat', 'lon'])
aod_1d = aod_1d.rename('aod_1d')
anth_aod_1d = anth_aod_1d.rename('anth_aod_1d')
natural_aod_1d = natural_aod_1d.rename('natural_aod_1d')
#%%
print('output will be saved into {}'.format(out_file_path))
ds = xr.merge([anth_aod_frac, natural_aod_frac, sulfate_anth_frac, sulfate_natural_frac,
               anth_aod, natural_aod,
               anth_aod_frac_1d, natural_aod_frac_1d,
               aod_1d, anth_aod_1d, natural_aod_1d
               ])
ds.to_netcdf(out_file_path, unlimited_dims='time')  # unlimited dim is there to allow ncrcat

#%%
