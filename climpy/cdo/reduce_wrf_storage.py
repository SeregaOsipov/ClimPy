from distutils.util import strtobool
import numpy as np
import xarray as xr
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import argparse
import climpy.utils.wrf_chem_utils as wrf_chem

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Script reduces the storage footprint of WRF output, configurable
Options:
 1. select boa level
 2. select vars

on Shaheen, interactive job
salloc --partition=shared --mem=150G
source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311
ipython


THOFA case:
FP_ROOT=/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v0
mkdir $FP_ROOT/pp
sbatch --job-name=reduce_wrf_storage --account=k10009 --time=8:00:00 --partition=shared --mem=150G --wrap="source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311; time python -u ${CLIMPY}/climpy/cdo/reduce_wrf_storage.py --select_boa=True --wrf_in=${FP_ROOT}/wrfout_d01_2023-*_00_00_00 --wrf_out=${FP_ROOT}/pp/wrfout_d01_thofa"

FP_ROOT=/scratch/osipovs/Data/AirQuality/THOFA/srs/39662
mkdir $FP_ROOT/pp
sbatch --job-name=reduce_wrf_storage --account=k10009 --time=8:00:00 --partition=shared --mem=150G --wrap="source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311; time python -u ${CLIMPY}/climpy/cdo/reduce_wrf_storage.py --select_boa=True --wrf_in=${FP_ROOT}/wrfout_d01_2023-*_00_00_00 --wrf_out=${FP_ROOT}/pp/wrfout_d01_thofa"
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path")#, default='/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/wrfout_d01_2017-12-14_00_00_00')
parser.add_argument("--wrf_out", help="wrf output file path")# , default='/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_wrf/wrfout_d01_2017-12-14_00_00_00')
parser.add_argument('--select_boa', type=strtobool, default=True)#required=True)
parser.add_argument('--vars', default='Times,no,no2,no3,hno3,hno4,ch4,co,co2,no,no2,o3,so2,PSFC,T2,eth,ete,hc3'.split(',')+wrf_chem.get_chemistry_package_definition(100), help='list of variables to extract')
parser.add_argument('--preload', type=strtobool, default=False)  # preloading make things slower
args = parser.parse_args()

#Debugging
# args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v0/wrfout_d01_2023-05-15_00_00_00'
# args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v0/pp/wrfout_d01_2023-05-15_00_00_00'
#
# args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v0/wrfout_d01_2023-*_00_00_00'
# args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v0/pp/wrfout_d01_thofa'

# args.vars='no,no2,no3,hno3,hno4,ch4,co,co2,no,no2,o3,so2,PSFC,T2,eth,ete,hc3'.split(',')+wrf_chem.get_chemistry_package_definition(100)

print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
print('select_boa: {}'.format(args.select_boa))
print('vars: {}'.format(args.vars))
#%%

if __name__ == '__main__':
    print('Starting Dask Client')
    from dask.distributed import Client
    client = Client(n_workers=25, threads_per_worker=1, memory_limit='5GB')  # levante
    print('Client is ready {}'.format(client))  # http://127.0.0.1:8787/status

    def preprocessing(ds):
        if args.vars:
            ds = ds[args.vars]  # selecting vars reduces footprint by almost factor of 2
        if args.select_boa:
            ds = ds.isel(bottom_top=0, bottom_top_stag=0, missing_dims='warn')
        return ds

    ds = xr.open_mfdataset(args.wrf_in, combine='nested', concat_dim='Time', parallel=True, preprocess=preprocessing)

    print(ds)
    if args.preload:
        print('Preloading Dataset')
        ds.load()  # this could speed up things
    print('Writing NetCDF to Disk')
    ds.to_netcdf(args.wrf_out)
    print('DONE')