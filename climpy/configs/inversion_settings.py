import xarray as xr
import subprocess
from climpy.utils.tropomi_utils import get_tropomi_species_configs, SENTINEL_DATA_ROOT_PATH
from types import SimpleNamespace
from pathlib import Path
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc

# BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = get_root_storage_path_on_hpc()
THOFA_D02_GRID_ID = 'THOFA_d02'

EMISSIONS_SRM_V1 = 'v1'

EMISSIONS_SRM_V_VOC_ALKANES = 'v_voc_alkanes'
ALKANE_SPECIES='E_ETH, E_HC3, E_HC5, E_HC8'.split(', ')

EMISSIONS_SRM_V_VOC_AROMATICS = 'v_voc_aromatics'
# AROMATICS_SPECIES='E_TOL, E_XYL, E_ETE, E_OLI, E_OLT, E_ALD, E_KET'.split(', ')
AROMATICS_SPECIES='E_TOL, E_XYL, E_OL2, E_OLI, E_OLT, E_ALD, E_KET'.split(', ') # E_OL2 is E_ETE


def build_wrf_config_from_template(wrf_scenario, inversion_version, tropomi_species_configs=None):
    wrf_grid_id = THOFA_D02_GRID_ID
    # base_dir = Path(get_root_storage_path_on_hpc())
    base_dir = Path('/scratch/osipovs/')  # temp fix
    inv_root = base_dir / 'Data/AirQuality/THOFA/inversion' / inversion_version

    config = SimpleNamespace(
        wrf_grid_id=wrf_grid_id, scenario=wrf_scenario, inversion_version=inversion_version,
        wrf_output_folder_path=inv_root / 'run_srs_{scenario}/'.format(scenario=wrf_scenario),
        wrf_tropomi_like_diags_folder_path_template='/scratch/osipovs/Data/AirQuality/THOFA/inversion/{version}/run_srs_{scenario}/pp/tropomi_like_{diag_key}/',
        tropomi_fp_template = SENTINEL_DATA_ROOT_PATH + '/{}/'.format(wrf_grid_id),
        tropomi_meta_data_folder_path='/scratch/osipovs/Data/AirQuality/THOFA/inversion/tropomi_meta_data/'
    )

    if tropomi_species_configs is not None:
        initialize_tropomi_section_of_wrf_config(config, tropomi_species_configs)

    return config


def initialize_tropomi_section_of_wrf_config(wrf_config, tropomi_species_configs):
    fps_dict = {}
    for tropomi_specie in tropomi_species_configs:
        fps_dict[tropomi_specie.diag_key] = wrf_config.wrf_output_folder_path / 'pp/tropomi_like_{diag_key}.nc'.format(diag_key=tropomi_specie.diag_key)
    wrf_config.wrf_tropomi_like_diag_fps = fps_dict

    fps_dict = {}
    for tropomi_specie in tropomi_species_configs:
        fps_dict[tropomi_specie.diag_key] = Path(wrf_config.tropomi_meta_data_folder_path) / 'tropomi_meta_{}.csv'.format(tropomi_specie.diag_key.lower())
    wrf_config.tropomi_meta_data_fps = fps_dict

    # # config.wrf_fp = config.wrf_tropomi_like_diags_folder_path_template.format(version=config.inversion_version, scenario=config.scenarios[0], diag_key=config.diag_key)  # wrf_tropomi_like_diag_fp
    # if getattr(wrf_config, 'wrf_tropomi_like_diags_folder_path_template', None) is None:
    #     wrf_config.wrf_tropomi_like_diags_folder_path_template = wrf_config.wrf_tropomi_like_diags_folder_path_template.format(version=wrf_config.inversion_version, scenario=wrf_config.scenario, diag_key=wrf_config.diag_key)  # folder with individual orbits
    # if getattr(wrf_config, 'wrf_tropomi_like_diags_fp', None) is None:
    #     wrf_config.wrf_tropomi_like_diags_fp = Path(wrf_config.wrf_output_folder_path) / 'pp/tropomi_like_{}.nc'.format(wrf_config.diag_key.lower())  # individual orbits merged into one file


def generate_inversion_config(inversion_version='v2', mapping_version='vTHOFA_TROPOMI', emissions_srs_version=EMISSIONS_SRM_V1):  # vETH
    base_dir = Path(get_root_storage_path_on_hpc())
    base_dir = Path('/scratch/osipovs/')  # temp fix
    print('Manually set base dir to /scratch/osipovs/')
    inv_root = base_dir / 'Data/AirQuality/THOFA/inversion'
    versioned_inv_root = inv_root / inversion_version / 'derivatives' / emissions_srs_version

    defaults_config = SimpleNamespace(
        inversion_version=inversion_version,
        mapping_version=mapping_version,
        emissions_srs_version=emissions_srs_version,  # version of perturbing emissions and quantyfing the SRS matrix

        # General
        geo_em_fp=base_dir / 'Data/AirQuality/THOFA/IC_BC/geo_em.d02.nc',
        sources_mapping_fp=base_dir / 'Data/AirQuality/THOFA/inversion/sources_mapping' / f'sources_mapping_d02_{mapping_version}.nc',

        # Emissions
        ei_ref_fp=base_dir / 'Data/AirQuality/THOFA/emissions/HERMESv3_radm2_madesorgam_20230515_THOFA_EDGARv81_GHG.nc_d02',
        ei_sens_fp_template=str(inv_root / f'emissions/{emissions_srs_version}/srs/' / 'HERMESv3_sens_{}.nc'),  # These emissions account all possible sources, potentially wider than the selected sources mapping
        ei_sens_ensemble_fp=versioned_inv_root / 'emissions_ensemble.nc',  # Ensemble of EI for all sensitivity experiments
        perturbed_sources_ensemble_fp=versioned_inv_root / 'perturbed_sources_ensemble.nc',  # dE by source, i.e. source ensemble ( in source-receptor notation)
        ei_revised_fp=versioned_inv_root / 'HERMESv3_radm2_madesorgam_20230515_THOFA_EDGARv81_GHG.nc_d02_revised',

        # Aux
        base_dir=base_dir, inv_root=inv_root, versioned_inv_root=versioned_inv_root,
    )

    return defaults_config


def get_ship_track_inversion_config(default_inversion_config):
    '''
    :param default_inversion_config: = get_default_inversion_config()
    :return:
    '''
    versioned_inv_root = default_inversion_config.versioned_inv_root

    ship_track_config = SimpleNamespace(
        # Matrices
        source_receptor_matrix_fp = versioned_inv_root / 'source_receptor_matrix_st.nc',
        # Receptors
        receptor_ref_fp = versioned_inv_root / 'run_srs_ref/pp/ship_track/wrf_ship_track.nc',
        receptor_sens_fp_template=str(versioned_inv_root / 'sensitivity_runs/run_srs_{ind}/pp/ship_track/wrf_ship_track.nc'),
        receptor_sens_ensemble_fp=versioned_inv_root / 'wrf_ship_track_ensemble.nc',

        inversion_cache_fp = versioned_inv_root / 'inversion_cache_ship_track.nc',
    )

    merged_config = SimpleNamespace(**vars(default_inversion_config), **vars(ship_track_config))
    return merged_config


def get_tropomi_like_inversion_config(tropomi_specie_config, default_inversion_config):
    versioned_inv_root  = default_inversion_config.versioned_inv_root

    tropomi_like_config = SimpleNamespace(
        # Matrices
        source_receptor_matrix_fp=versioned_inv_root / 'source_receptor_matrix_tropomi_like_{}.nc'.format(tropomi_specie_config.diag_key),
        # Receptors
        receptor_ref_fp = versioned_inv_root / 'run_srs_ref/pp/tropomi_like_{}.nc'.format(tropomi_specie_config.diag_key),
        receptor_sens_fp_template=str(versioned_inv_root / 'sensitivity_runs/run_srs_{ind}/pp/tropomi_like_{key}.nc'),
        receptor_sens_ensemble_fp = versioned_inv_root / 'tropomi_like_{}_ensemble.nc'.format(tropomi_specie_config.diag_key),

        inversion_cache_fp = versioned_inv_root / 'inversion_cache_tropomi_like_{}.nc'.format(tropomi_specie_config.diag_key),
        ng_iteration_inversion_cache_fp=versioned_inv_root / 'inversion_cache_tropomi_like_{}_ng_i2.nc'.format(tropomi_specie_config.diag_key),  # TODO need to homogenize and generalize
    )

    merged_config = SimpleNamespace(**vars(default_inversion_config), **vars(tropomi_like_config))
    return merged_config


def get_latest_version_of_inversion_configs(inversion_version='v2', mapping_version='vTHOFA_TROPOMI', wrf_scenario='ref', emissions_srs_version=None):  # Utility function that combines the latest setup
    # get sources mapping
    default_inversion_config = generate_inversion_config(inversion_version, mapping_version, emissions_srs_version)
    sources_mapping_ds = get_sources_mapping_ds(default_inversion_config)

    tropomi_species_configs = get_tropomi_species_configs()
    wrf_config = build_wrf_config_from_template(wrf_scenario, inversion_version, tropomi_species_configs)

    tropomi_like_inversion_configs = []
    for tropomi_specie_config in tropomi_species_configs:  # ch4, co , etc
        tropomi_inversion_config = get_tropomi_like_inversion_config(tropomi_specie_config, default_inversion_config)
        config = SimpleNamespace(**{**vars(wrf_config), **vars(tropomi_specie_config), **vars(tropomi_inversion_config)})
        tropomi_like_inversion_configs.append(config)

    # Separate inversion config for Ship Track diagnostics
    thofa_inversion_config = get_ship_track_inversion_config(default_inversion_config)
    # inversion_configs.append(thofa_inversion_config)

    return sources_mapping_ds, thofa_inversion_config, tropomi_like_inversion_configs, tropomi_species_configs


def build_sensitivity_ensemble_file_paths(config, sources_mapping_ds, tropomi_like_key=None):
    '''
    tropomi_like_key is tropomi_specie_config.diag_key unless ship track case
    '''
    receptor_sens_fps = []  # List of sensitivity experiments
    emissions_sens_fps = []

    for flat_index, lat, lon in zip(sources_mapping_ds.source_index_1d, sources_mapping_ds.lat_1d, sources_mapping_ds.lon_1d):
        receptor_sens_fps.append(config.receptor_sens_fp_template.format(ind=int(flat_index), key=tropomi_like_key))
        emissions_sens_fps.append(config.ei_sens_fp_template.format(int(flat_index)))

    config.receptor_sens_fps = receptor_sens_fps
    config.emissions_sens_fps = emissions_sens_fps


def get_sources_mapping_ds(config):
    '''
    Exclude faulty sources

    import pandas as pd
    df = pd.read_csv('/scratch/osipovs/Models/WRF_run/job_array_indices_rosenbrock.csv', sep=':', header=None)
    sources_to_exclude = list(df[1])
    sources_to_exclude = [1414, 1937, 1964, 2384, 2820, 3835, 7135, 7247, 7902, 7949, 8130, 9513, 9643, 9820]  # these are temporarily bad
    sources_mapping_ds = sources_mapping_ds.drop_sel(source_index_1d=sources_to_exclude)

    :param config:
    :return:
    '''
    sources_mapping_ds = xr.open_dataset(config.sources_mapping_fp)
    if 'n' in sources_mapping_ds.dims:  # old style of naming
        sources_mapping_ds = sources_mapping_ds.rename_dims({'n': 'source_index_1d'})

    # TODO: cast to int. This should be fixed at nc creation time
    sources_mapping_ds['source_index_1d'] = sources_mapping_ds.source_index_1d.astype(int)

    return sources_mapping_ds


def inject_sources_mapping_info(ds, sources_mapping_ds):
    ds = ds.assign_coords({
        "source_index_1d": sources_mapping_ds.source_index_1d,
        "lat_1d": sources_mapping_ds.lat_1d,
        "lon_1d": sources_mapping_ds.lon_1d
    })
    ds = ds.set_index(source_index_1d='source_index_1d')

    return ds


def execute_nco_command(command):
    try:
        # print(f"Executing: {command}")
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("!!! NCO FAILED !!!")
        print(f"Exit Code: {e.returncode}")
        print("--- ERROR LOG ---")
        print(e.stderr)  # <--- THIS IS THE CRITICAL INFO
        print("-----------------")


def create_ensembles_merging_individual_files(sources_mapping_ds, tropomi_like_inversion_configs, inversion_config_for_ship_track=None, handle_difference_in_var_sets=False):
    print('Merging Ensembles from individual files')

    inversion_config = tropomi_like_inversion_configs[0]
    build_sensitivity_ensemble_file_paths(inversion_config, sources_mapping_ds)

    if inversion_config_for_ship_track is not None:
        inversion_config = inversion_config_for_ship_track
        print("\nMerging Ship Track Ensemble: {} files\n".format(len(inversion_config.receptor_sens_fps)))

        nco_command_base = ["ncecat", "-O", "-u", "source_index_1d"]
        if handle_difference_in_var_sets:  # This is the case with difference set of variable in wrf output files
            # slim_file = inversion_config_for_ship_track.receptor_sens_fps[-7]  # file 12212
            # heavy_file = inversion_config_for_ship_track.receptor_sens_fps[-6]  # any other
            # print('Slim file: {}'.format(slim_file))
            # slim_vars = set(xr.open_dataset(slim_file).data_vars.keys())
            # heavy_vars = set(xr.open_dataset(heavy_file).data_vars.keys())
            # common_vars_set = slim_vars.intersection(heavy_vars)

            # Generic approach, check all files
            var_sets = [set(xr.open_dataset(fp).data_vars) for fp in inversion_config_for_ship_track.receptor_sens_fps]
            common_vars_set = set.intersection(*var_sets)
            common_vars = ','.join(common_vars_set)
            print('Common vars: {}'.format(common_vars))
            # nco_command_base = ["ncecat", "-4", "-L", "1", "-O", "--no_tmp_fl", "-v", common_vars, "-u", "source_index_1d"]
            nco_command_base = ["ncecat", "-O", "--no_tmp_fl", "-v", common_vars, "-u", "source_index_1d"]

        nco_command = nco_command_base + inversion_config.receptor_sens_fps + [inversion_config.receptor_sens_ensemble_fp]
        execute_nco_command(nco_command)
        print('Done: {}'.format(inversion_config.receptor_sens_ensemble_fp))


    nco_command_base = ["ncecat", "-O", "-u", "source_index_1d"]

    print('\nMerging Emissions Ensemble: {} files\n'.format(len(inversion_config.emissions_sens_fps)))
    # Emissions Ensemble
    nco_command = nco_command_base + inversion_config.emissions_sens_fps + [str(inversion_config.ei_sens_ensemble_fp)]
    execute_nco_command(nco_command)
    print('Done: {}'.format(inversion_config.ei_sens_ensemble_fp))

    print('\nMerging Tropomi Ensembles')
    for inversion_config in tropomi_like_inversion_configs:
        build_sensitivity_ensemble_file_paths(inversion_config, sources_mapping_ds, inversion_config.diag_key)
        print('{}: {} files'.format(inversion_config.diag_key, len(inversion_config.receptor_sens_fps)))
        nco_command = nco_command_base + inversion_config.receptor_sens_fps + [inversion_config.receptor_sens_ensemble_fp]
        execute_nco_command(nco_command)
        print('Done: {}'.format(inversion_config.receptor_sens_ensemble_fp))

    #     display(getattr(inversion_config, 'tropomi_like_{}_sens_ensemble_fp'.format(specie_config.diag_key)))
    #     nco_command = nco_command_base + getattr(inversion_config, 'tropomi_like_{}_sens_fps'.format(specie_config.diag_key)) + [getattr(inversion_config, 'tropomi_like_{}_sens_ensemble_fp'.format(specie_config.diag_key))]
    #     execute_nco_command(nco_command)

    print('Done')