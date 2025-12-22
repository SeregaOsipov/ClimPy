#!/bin/bash -l
#SBATCH --job-name=wrf_health_pp
#SBATCH --account=k10009  # k10048
#SBATCH --partition=workq
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=192
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=Sergey.Osipov@kaust.edu.sa

source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311
#module load nco cdo # On Shaheen 3 nco and cdo needs to be provided from python environment

# run example:
# scenario=EDGAR_trend
# sbatch $CLIMPY/climpy/wrf/wrf_optics_pp.sh /work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/ true true false
#for scenario in EDGAR_trend CLE MFR MFR_LV
#do
#    echo $scenario
#    sbatch $CLIMPY/climpy/wrf/wrf_optics_pp.sh /work/mm0062/b302074/Data/AirQuality/EMME/2050/$scenario/chem_100_v1/output/ false true
#done

#sim_version="chem_100_v22"
#sim_version=$1
wrf_output_folder_path=$1  # /work/mm0062/b302074/Data/AirQuality/AQABA/${sim_version}/output
do_cdo=${2:-false}
do_python=${3:-false}
do_on_modis_grid=${4:-false}

echo "Starting wrf_optics_pp.sh for {$wrf_output_folder_path}"

if [ ! -d ${wrf_output_folder_path} ]; then
  echo "The wrf_output_folder_path directory does not exist: ${wrf_output_folder_path}"
  exit 1
fi

#if [ "$do_cdo" = true ] ; then
#  # create log dir
#  mkdir -p ${wrf_output_folder_path}/pp_optics/logs
#  mkdir -p ${wrf_output_folder_path}/pp_optics/merge  # for a single file with all vars merged
#
#  # full list
#  vars=(ALT PH PHB H2OAI H2OAJ so4aj so4ai nh4aj nh4ai no3aj no3ai naaj naai claj clai asoa1j asoa1i asoa2j asoa2i asoa3j asoa3i asoa4j asoa4i bsoa1j bsoa1i bsoa2j bsoa2i bsoa3j bsoa3i bsoa4j bsoa4i orgpaj orgpai ecj eci caaj caai kaj kai mgaj mgai p25j p25i antha seas soila nu0 ac0 corn NU3 AC3 COR3)
#
#  N=64  # control how many jobs to run in parallel
#  for var in ${vars[@]} ; do
#    ((i=i%N)); ((i++==0)) && wait
#    echo 'next var: ' ${var}
#    time cdo -O monmean -mergetime [ -select,name=${var} ${wrf_output_folder_path}/wrfout_d01_*_00_00_00 ] ${wrf_output_folder_path}/pp_optics/wrfout_d01_monmean_${var} >& ${wrf_output_folder_path}/pp_optics/logs/log.wrfout_d01_monmean_${var} &
#  done
#
#  wait
#  echo "DONE monmean, start merge"
#
#  # -O means overwrite existing file & extract one full year [July 2017-July 2018)
#  #time cdo -O seltimestep,2/13 -merge ${wrf_output_folder_path}/pp_optics/wrfout_d01_monmean_* ${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean
#  time cdo -O -merge ${wrf_output_folder_path}/pp_optics/wrfout_d01_monmean_* ${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean
#  time cdo -O timmean ${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean ${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_timmean
#fi

if [ "$do_python" = true ] ; then
  gogomamba
#  python -u ${CLIMPY}/climpy/wrf/wrf_pp_optics.py --wrf_in=${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean --wrf_out=${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean_optics >& ${wrf_output_folder_path}/pp_optics/logs/log.optics_python
  python -u ${CLIMPY}/climpy/wrf/wrf_pp_optics.py --wrf_in=${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_timmean --wrf_out=${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_timmean_optics >& ${wrf_output_folder_path}/pp_optics/logs/log.optics_python
fi

# regrid on MODIS grid
if [ "$do_on_modis_grid" = true ] ; then
  gogomamba

  python -u ${CLIMATE_MODELLING}/Papers/AQABA/paper_figures/regrid_wrf_output_on_modis_grid.py --wrf_in=${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean --wrf_out=${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean_regrid >& pp_optics/logs/log.regrid_python
  echo "DONE regrid"

  python -u ${CLIMPY}/climpy/wrf/wrf_pp_optics.py --wrf_in=${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean_regrid --wrf_out=${wrf_output_folder_path}/pp_optics/merge/wrfout_d01_monmean_regrid_optics >& pp_optics/logs/log.optics_python
  echo "DONE pp optics"

  python -u ${CLIMATE_MODELLING}/Papers/AQABA/paper_figures/modis_aod_figure.py --sim_version=${sim_version} >& pp_optics/logs/log.modis_aod_figure_python
  echo "DONE merge"
fi
