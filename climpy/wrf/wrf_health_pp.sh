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
#for scenario in 2017 2050/HLT 2050/CLE 2050/MFR 2050/MFR_LV
#do
#    echo $scenario
#    sbatch $CLIMPY/climpy/wrf/wrf_health_pp.sh /work/mm0062/b302074/Data/AirQuality/EMME/$scenario/chem_100_v1/output/ false true
#done

#sbatch $CLIMPY/climpy/wrf/wrf_health_pp.sh /scratch/osipovs/Data/AirQuality/AREAD/chem_100_v2/output false true

wrf_output_folder_path=$1  # /work/mm0062/b302074/Data/AirQuality/AQABA/${sim_version}/output
do_boa=${2:-true}
do_python_health=${3:-true}

echo "do_boa is $do_boa"
echo "do_python_health is $do_python_health"
echo "Starting wrf_health_pp.sh for ${wrf_output_folder_path}"

if [ ! -d ${wrf_output_folder_path} ]; then
  echo "The wrf_output_folder_path directory does not exist: ${wrf_output_folder_path}"
  exit 1
fi

cd ${wrf_output_folder_path}
cwd=$(pwd)

if [ "$do_boa" = true ] ; then
  mkdir -p ${wrf_output_folder_path}/boa/logs

#  module load cdo
  echo "extract BOA using CDO"
  N=96  # control how many jobs to run in parallel
  for file in wrfout*00; do
    ((i=i%N)); ((i++==0)) && wait
    echo ${file}
    logfile=${file}.log
#    cdo sellevidx,1 -select,name=T,T2,Q2,o3,so2,no,no2,co,ALT,H2OAI,H2OAJ,so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,asoa1j,asoa1i,asoa2j,asoa2i,asoa3j,asoa3i,asoa4j,asoa4i,bsoa1j,bsoa1i,bsoa2j,bsoa2i,bsoa3j,bsoa3i,bsoa4j,bsoa4i,orgpaj,orgpai,ecj,eci,caaj,caai,kaj,kai,mgaj,mgai,p25j,p25i,antha,seas,soila,nu0,ac0,corn,NU3,AC3,COR3,pan,hcho,POTENTIAL_OF_HYDROGEN_PH,PH_ERYTHEMA ./${file} ./boa/${file} >& boa/logs/${logfile}_cdo &
    cdo sellevidx,1 ./${file} ./boa/${file} >& boa/logs/${logfile}_cdo &
  done
  wait
#  module unload cdo
  echo "done do_boa section"
fi

if [ "$do_python_health" = true ] ; then  # work with daily averages
  mkdir -p ${wrf_output_folder_path}/pp_health/logs

  echo "derive PMs via python"
  N=64 #  64  # 32  # control how many jobs to run in parallel  # 32 sometimes crashes

  for file in wrfout*00; do  # cdo/daily/
    ((i=i%N)); ((i++==0)) && wait
    echo ${file}
    logfile=${file}.log
    python -u ${CLIMPY}/climpy/wrf/wrf_pp_health.py --wrf_in=${cwd}/${file} --wrf_out=${cwd}/pp_health/${file} >& pp_health/logs/${logfile}_python &
  done
  wait
  echo "done do_python_health section"
fi

echo "DONE pp_health.sh"

exit 0


