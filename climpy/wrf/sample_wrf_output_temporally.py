import xarray as xr
import argparse
import pandas as pd

'''
Sample WRF output temporally

storage_path=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v4/run_srs_ref/pp/column/
storage_path=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_revised/pp/column/
python -u ${CLIMPY}/climpy/wrf/sample_wrf_output_temporally.py --wrf_in_file_path=${storage_path}/wrfout* --wrf_out_file_path=${storage_path}/wrf_sampled_over_tropomi.nc --date_csv_fp=/project/k10048/osipovs/Data/AirQuality/THOFA/tropomi_dates.csv 

storage_path=/scratch/osipovs//Data/AirQuality/THOFA/inversion/v5/
python -u ${CLIMPY}/climpy/wrf/sample_wrf_output_temporally.py --wrf_in_file_path=$storage_path}/source_receptor_matrix_ca.nc --wrf_out_file_path=${storage_path}/source_receptor_matrix_ca_sampled_over_tropomi.nc --date_csv_fp=/project/k10048/osipovs/Data/AirQuality/THOFA/tropomi_dates.csv
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="the are only to support pycharm debugging")
parser.add_argument("--wrf_in_file_path", help="wrfout file path as input", required=True, default='/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_revised/pp/column/wrfout*')
parser.add_argument("--wrf_out_file_path", help="pp wrf file path as output", required=True, default='/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_revised/pp/column/wrf_sampled_over_tropomi.nc')
parser.add_argument("--date_csv_fp", help="file with Dates to sample", required=True, default='/project/k10048/osipovs/Data/AirQuality/THOFA/tropomi_dates.csv')  # required=True)
args = parser.parse_args()

print('Will process WRF:\nin {}\nout {}'.format(args.wrf_in_file_path, args.wrf_out_file_path))
print('Dates will be sampled from:\n{}'.format(args.date_csv_fp))
#%%
ds = xr.open_mfdataset(args.wrf_in_file_path, concat_dim="Time", combine="nested", parallel=True)
df = pd.read_csv(args.date_csv_fp, index_col=0, parse_dates=True)

min_wrf_date = pd.to_datetime(ds.Time.min().item())
max_wrf_date = pd.to_datetime(ds.Time.max().item())
print('Reducing CSV coverage\n{} : {}\n to WRF coverage\n{} : {}'.format(df.index.min(), df.index.max(), min_wrf_date, max_wrf_date))
df = df.loc[min_wrf_date:max_wrf_date]

sampled_ds = ds.interp(Time=df.index)
sampled_ds.to_netcdf(args.wrf_out_file_path)