# %load_ext autoreload
# %autoreload 2
from distutils.util import strtobool

from climpy.configs.inversion_settings import initialize_tropomi_section_of_wrf_config
# Fix "display" issue in terminal
from climpy.utils.common_utils import regularize_display_in_terminal

regularize_display_in_terminal()

import os
from datetime import datetime
from types import SimpleNamespace
from climpy.utils.tropomi_utils import regrid_tropomi_on_wrf_grid_in_batch, get_tropomi_species_configs, download_tropomi, prepare_tropomi_metadata, process_metadata_deriving_tropomi_like_diags
import argparse
from pathlib import Path
import subprocess

'''
The goal:
Given the folder with WRF output, download TROPOMI data and calculate corresponding TROPOMI-like diagnostics

python $CLIMPY/climpy/wrf/derive_tropomi_like_diags_for_wrf.py --wrf_output_folder_path=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/sensitivity_runs/run_srs_9964 --tropomi_meta_data_folder_path=/scratch/osipovs/Data/AirQuality/THOFA/inversion/tropomi_meta_data

To configure species:  --species=chocho

Compute node on Shaheen3

sbatch --job-name=derive_tropomi_like_diags_for_wrf --account=k10009 --time=24:00:00 --partition=workq -N 1 --ntasks=192 --ntasks-per-node=192 --hint=nomultithread <<'EOT'
#!/bin/bash -l
source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311;
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
wrf_dir=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v2/run_srs_revised
wrf_dir=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v2/derivatives/v1/run_srs_revised_ng_i2
python -u $CLIMPY/climpy/wrf/derive_tropomi_like_diags_for_wrf.py --wrf_output_folder_path=${wrf_dir} --tropomi_meta_data_folder_path=/scratch/osipovs/Data/AirQuality/THOFA/inversion/tropomi_meta_data --overwrite_existing_files=True
EOT


'''
parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="pycharm")
parser.add_argument("--wrf_output_folder_path", help="wrf input file path")
parser.add_argument("--tropomi_meta_data_folder_path", help="Overrides default path with metadata files. Normally, this is derived given wrf output folder path")  # but on Shaheen, with no internet, I have to prefetch it
parser.add_argument("--overwrite_existing_files", type=strtobool, default=True)
parser.add_argument("--species", nargs='+', help="List of species to process (e.g., ch4 no2 co)", default=None)
args = parser.parse_args()

# args.wrf_output_folder_path='/scratch/osipovs/Data/AirQuality/THOFA/inversion/v2/run_srs_ref'
# args.tropomi_meta_data_folder_path='/scratch/osipovs/Data/AirQuality/THOFA/inversion/tropomi_meta_data'

### Filter which species to process
all_species_configs = get_tropomi_species_configs()

# 2. Filter configs based on user input
if args.species:
    print('Species Filter: {}'.format(args.species))
    # Normalize input to lowercase to match diag_key
    selected_species = [s.lower() for s in args.species]
    tropomi_species_configs = [cfg for cfg in all_species_configs if cfg.diag_key in selected_species]

    # Check if any requested species were not found
    found_keys = [cfg.diag_key for cfg in tropomi_species_configs]
    for s in selected_species:
        if s not in found_keys:
            raise Exception(f"Warning: Species '{s}' not recognized. Available: {found_keys}")
else:
    tropomi_species_configs = all_species_configs

#### COntinue with initialization

wrf_config = SimpleNamespace(
    wrf_grid_id='THOFA_d02',
    wrf_output_folder_path = Path(args.wrf_output_folder_path),
    tropomi_meta_data_folder_path = Path(args.tropomi_meta_data_folder_path),
    wrf_filter_dates = [datetime(2023, 6, 1), datetime(2023, 6, 25)]  # last wrfout file is not full. Tropomi-like diags break
)
initialize_tropomi_section_of_wrf_config(wrf_config, tropomi_species_configs)

configs = []
for specie_settings in tropomi_species_configs:  # ch4, co , etc
    config = SimpleNamespace(**{**vars(wrf_config), **vars(specie_settings)})
    configs.append(config)

aggregate_individual_tropomi_like_files = True  # concatenate in time individual tropomi-like files
#%% Main loop

for config in configs:
    print('Processing config: {}'.format(config.diag_key))
    display(config)

    #%% check if the result exists already
    time_aggregated_diag_fp = config.wrf_tropomi_like_diag_fps[config.diag_key]
    if not args.overwrite_existing_files and aggregate_individual_tropomi_like_files and time_aggregated_diag_fp.exists():
        print('Skipping. Result already exists: {}'.format(config.wrf_tropomi_like_diag_fps[config.diag_key]))
        continue

    #%% derive_tropomi_like_diags
    meta_df = prepare_tropomi_metadata(config)
    need_to_download_tropomi = False  # avoid running multiple download sessions in parallel
    if need_to_download_tropomi:
        download_tropomi_impl = download_tropomi  # default case
        if config.diag_key=='chocho':  # special case
            download_tropomi_impl = download_tropomi_chocho_from_meta
        download_tropomi_impl(meta_df)
    regrid_tropomi_on_wrf_grid_in_batch(meta_df, config.wrf_grid_id, config.tropomi_key)
    # derive_information_fraction(meta_df, config.tropomi_key, config.wrf_grid_id)  # Derive List of TROPOMI files with good coverage
    process_metadata_deriving_tropomi_like_diags(meta_df, config)  # Process MetaData list, Deriving TROPOMI-like diagnostics for WRF output

    # The main logic is done here. Next is only PPing
    #%% merge files into one
    if aggregate_individual_tropomi_like_files:
        print('Merging individual files into one')
        processed_files = meta_df.wrf_tropomi_like_diag_fp.tolist()

        # xarray is too slow for ncrcat
        # ds_combined = xr.open_mfdataset(processed_files, combine='nested', concat_dim='time', parallel=True)
        # datasets = [xr.open_dataset(f).load() for f in processed_files]
        # ds_combined = xr.concat(datasets, dim='time')
        # ds_combined.to_netcdf(diag_fp)

        # use ncrcat instead
        subprocess.run(['ncrcat', '-O'] + processed_files + ['-o', time_aggregated_diag_fp], check=True)
        print('Saved to {}'.format(time_aggregated_diag_fp))

        delete_intermediate_files = True
        if delete_intermediate_files:  # remove individual files
            for fp in processed_files:
                os.remove(fp)

            if processed_files:
                intermediate_folder = Path(processed_files[0]).parent
                os.rmdir(intermediate_folder)  # deleting folder is riskier. rmdir only deletes empty directories

            ## tropomi_like_diag_folder = os.path.dirname(processed_files[0])
            # tropomi_like_diag_folder = config.wrf_output_folder_path + 'pp/tropomi_like_{}/'.format(config.diag_key.lower())
            # print('Removing intermediate folder: {}'.format(tropomi_like_diag_folder))  # This is quite dangerous
            # shutil.rmtree(tropomi_like_diag_folder, ignore_errors=True)

print('derive_tropomi_like_diags_for_wrf.py completed successfully.')