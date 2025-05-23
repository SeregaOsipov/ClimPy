import numpy as np
import xarray as xr
import argparse
from climpy.utils.wrf_chem_made_utils import get_wrf_size_distribution_by_modes
from climpy.utils.refractive_index_utils import mix_refractive_index
import climpy.utils.mie_utils as mie
from distutils.util import strtobool

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
This script derives optical properties from WRF output 
Currently only column AOD for MADE only (log-normal pdfs)

Ackermann MADE paper: https://www.sciencedirect.com/science/article/pii/S1352231098000065

run EMME AQ projection sims suite:

# 2017
sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/cdo/wrfout_d01_timmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_optics/wrfout_d01_timmean_optics_by_mode"
sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=True --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/cdo/wrfout_d01_timmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_optics/wrfout_d01_timmean_optics"

sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/cdo/wrfout_d01_2017-01_monmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_optics/wrfout_d01_2017-01_monmean_optics_by_mode"
sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/cdo/wrfout_d01_2017-04_monmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_optics/wrfout_d01_2017-04_monmean_optics_by_mode"
sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/cdo/wrfout_d01_2017-07_monmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_optics/wrfout_d01_2017-07_monmean_optics_by_mode"
sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/cdo/wrfout_d01_2017-10_monmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_optics/wrfout_d01_2017-10_monmean_optics_by_mode"

#2050
for scenario in HLT CLE MFR MFR_LV
do
    echo $scenario
    mkdir -p /work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/pp_optics

    sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/cdo/wrfout_d01_timmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/pp_optics/wrfout_d01_timmean_optics_by_mode"
    sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=True --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/cdo/wrfout_d01_timmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/pp_optics/wrfout_d01_timmean_optics"
    
    sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/cdo/wrfout_d01_2050-01_monmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/pp_optics/wrfout_d01_2050-01_monmean_optics_by_mode"
    sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/cdo/wrfout_d01_2050-04_monmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/pp_optics/wrfout_d01_2050-04_monmean_optics_by_mode"
    sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/cdo/wrfout_d01_2050-07_monmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/pp_optics/wrfout_d01_2050-07_monmean_optics_by_mode"
    sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/cdo/wrfout_d01_2050-10_monmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/pp_optics/wrfout_d01_2050-10_monmean_optics_by_mode"
done


run examples
levante:
add --constraint=512G if you need more memory
yearly
sbatch --job-name=wrf_optics_pp --account=mm0062 --time=8:00:00 --partition=compute -N 1 --wrap="source ~/.bashrc; gogomamba; python -u ${CLIMPY}climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean_optics_by_mode"

local:
python -u  ${CLIMPY}/climpy/wrf/wrf_pp_optics.py --sum_up_modes=True --wrf_in=/home/osipovs/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean --wrf_out=/home/osipovs/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean_optics
python -u  ${CLIMPY}/climpy/wrf/wrf_pp_optics.py --sum_up_modes=False --wrf_in=/home/osipovs/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean --wrf_out=/home/osipovs/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean_optics_by_mode
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path")#, required=True)
parser.add_argument("--wrf_out", help="wrf output file path")#, required=True)
parser.add_argument("--sum_up_modes", help="True/False, sum up aerosol ijk modes", type=strtobool, default=True) #required=True)#default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v23/output/pp_wrf_optics/merge/wrfout_d01_monmean_regrid_optics')
args = parser.parse_args()

#DEBUG
# args.sum_up_modes = False
#local
# annual

args.wrf_in='/HDD2/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/cdo//wrfout_d01_timmean'
args.wrf_out = '/HDD2/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/wrfout_d01_timmean_optics_by_mode'
args.sum_up_modes=False
# monmean
# args.wrf_in='/HDD2/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_monmean'
# args.wrf_out = '/HDD2/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_monmean_optics'
# levante
# args.wrf_in = '/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean'
# args.wrf_out = '/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean_optics'

chem_opt = 100
# setup wl grid for optical props
default_wrf_wavelengths = np.array([0.3, 0.4, 0.5, 0.6, 0.999])  # default WRF wavelengths in SW
maritime_wavelengths = np.array([380, 440, 500, 675, 870]) / 10 ** 3  # aeronet
wavelengths = np.unique(np.append(default_wrf_wavelengths, maritime_wavelengths))  # Comprehensive wl grid

wavelengths = np.array([0.55, 10])  # SW & LW

#%% dask implementation


def calculate_mie(ri_ds):
    mie_das = xr.apply_ufunc(mie.get_mie_efficiencies_numpy, ri_ds.chunk({'time': 1, "south_north": 45, 'west_east': 45}).ri,
                             dN_ds['radius'], ri_ds['wavelength'],
                             input_core_dims=[["wavelength"], ["radius"], ['wavelength']],
                             output_core_dims=[["wavelength", 'radius'], ["wavelength", 'radius'],
                                               ["wavelength", 'radius'], ["wavelength", 'radius'],
                                               ["wavelength", 'radius', 'angle'], ['angle']],
                             vectorize=True,
                             dask='parallelized',
                             output_sizes={'angle': 180},  # default size of angles are not provided
                             output_dtypes=[ri_ds.wavelength.dtype, ri_ds.wavelength.dtype, ri_ds.wavelength.dtype,
                                            ri_ds.wavelength.dtype, ri_ds.wavelength.dtype, ri_ds.wavelength.dtype])

    var_names = 'qext, qsca, g, qasm, phase_function, phase_function_angles_in_radians'.split(', ')
    for key, da in zip(var_names, mie_das):
        da.name = key

    mie_ds = xr.merge(mie_das)
    mie_ds.load()

    return mie_ds


#%%
if __name__ == '__main__':
    # %%
    print('Starting Dask Client')
    from dask.distributed import Client

    # client = Client(n_workers=10, threads_per_worker=1, memory_limit='10GB')  # Local. memory_limit is per worker
    client = Client(n_workers=50, threads_per_worker=1, memory_limit='5GB')  # levante
    print('Client is ready {}'.format(client))  # http://127.0.0.1:8787/status
    #%%
    print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
    xr_in = xr.open_dataset(args.wrf_in, chunks={'XTIME': 1, "south_north": 45, 'west_east': 45})
    xr_in = xr_in.rename({'XTIME': 'time', 'XLAT': 'lat', 'XLONG': 'lon'})  # , 'west_east':'lon', 'south_north':'lat'})
    # %% Derive spectral column AOD
    ri_ds = mix_refractive_index(xr_in, chem_opt, wavelengths)  # volume weighted RI
    # ri_ds = ri_ds.rename({'XTIME': 'time', 'XLAT':'lat', 'XLONG':'lon', 'west_east':'lon', 'south_north':'lat'})
    # dA_ds = get_wrf_size_distribution_by_modes(xr_in, moment='dA', sum_up_modes=True, column=True)
    # dA_ds = dA_ds.rename({'XTIME': 'time', 'XLAT':'lat', 'XLONG':'lon'})
    dN_ds = get_wrf_size_distribution_by_modes(xr_in, moment='dN', sum_up_modes=args.sum_up_modes, column=True)
    # dV_ds = get_wrf_size_distribution_by_modes(xr_in, moment='dV', sum_up_modes=True, column=True)
    # dN_ds_by_modes = get_wrf_size_distribution_by_modes(xr_in, sum_up_modes=False, column=True)
    print('SD is ready')
    #%% get mie and integrate over SD
    mie_ds = calculate_mie(ri_ds)
    # dN_ds = dN_ds.chunk({"south_north": 45, 'west_east': 45})
    op_ds = mie.integrate_mie_over_aerosol_size_distribution(mie_ds, dN_ds, include_phase_function=False)  # phase function has very large memory footprint
    op_ds.to_netcdf(args.wrf_out)
    print('DONE: {}'.format(args.wrf_out))
#%% debugging subset
# dsize=15
# ri_ds = ri_ds.isel(south_north=slice(0,dsize), west_east=slice(0,dsize))
# dA_ds = dA_ds.isel(south_north=slice(0,dsize), west_east=slice(0,dsize))
# dN_ds = dN_ds.isel(south_north=slice(0,dsize), west_east=slice(0,dsize))
#%% debugging plots
# import matplotlib.pyplot as plt
# plt.ion()
# plt.figure()
# np.abs(ri_ds.isel(wavelength=0).ri.imag).plot(cmap='Oranges', vmax=0.05)
#
# op_ds.ssa
#
# mie_ds
# dN_ds
#
# mie_ds.qsca.isel(time=0, south_north=250, west_east=250, wavelength=1).plot()
# mie_ds.qext.isel(time=0, south_north=250, west_east=250, wavelength=1).plot()
#
# ssa = mie_ds.qsca / mie_ds.qext
#
# plt.figure()
# plt.clf()
# ssa.isel(time=0, south_north=250, west_east=250, wavelength=1).plot()
# plt.yscale('log')
# plt.xscale('log')