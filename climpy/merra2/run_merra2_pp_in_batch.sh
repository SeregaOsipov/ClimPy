#!/bin/bash  -l

# prep python environment
gogomamba

# slightly malformed input data
input_start=1991-6-15
input_end=1991-7-15

# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd
startdate=$(date -I -d "$input_start") || exit -1
enddate=$(date -I -d "$input_end")     || exit -1

N=32  # control how many jobs to run in parallel
d="$startdate"
while [ "$d" != "$enddate" ]; do
  ((i=i%N)); ((i++==0)) && wait  # run in parallel
  echo "Calling python script for date $d"
  python ${CLIMATE_MODELLING}/Papers/Zittis/merra2_aod_anth_natural_pp.py --date=$d &
  d=$(date -I -d "$d + 1 day")
done

wait
echo "DONE"