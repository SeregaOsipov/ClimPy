#!/bin/bash -l
#SBATCH --job-name=wps_unnetcdf_merra2_in_parallel
#SBATCH --partition=workq  # shared
#SBATCH --account=k10009
##SBATCH --ntasks=50
#SBATCH --hint=nomultithread
#SBATCH --time=8:00:00
##SBATCH --mail-type=end         # send email if job fails
#SBATCH --mail-user=Sergey.Osipov@kaust.edu.sa

# Run example:
# sbatch $CLIMPY/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_merra2_in_parallel.sh

source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311

echo $pwd
echo "wps_unnetcdf_merra2_in_parallel"

out_storage_path=/scratch/osipovs/Data/NASA/MERRA2/unnetcdf/
START_DATE="2021-01-01"
END_DATE="2022-01-01"
CHUNK_START=$(date -d "${START_DATE}" +%Y-%m-%d)
INDEX=1

while [[ "$(date -d "${CHUNK_START}" +%s)" -lt "$(date -d "${END_DATE}" +%s)" ]]; do
    CHUNK_END=$(date -d "${CHUNK_START} + 1 week" +%Y-%m-%d)  # control chunks here: month, week, day
    echo $INDEX
    echo $CHUNK_START
    echo $CHUNK_END

    logfile=log.unnetcdf_chunk_${INDEX}
    echo ${logfile}

    python -u ${CLIMPY}/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_merra2.py --start_date=${CHUNK_START} --end_date=${CHUNK_END} --out_storage_path=${out_storage_path} >& ${logfile} &

    # The next chunk starts where this one ended.
    CHUNK_START="${CHUNK_END}"
    INDEX=$((INDEX + 1))
done

wait
echo "DONE"

