#!/bin/bash
FILE=/array1/Reanalysis/ERA-Interim/msl.1979-2018.075.nc
VAR=msl
if [ "$1" != "" ]; then
  mpirun -np $1 python ../pystormtracker/stormtracker.py -i ${FILE} -v ${VAR} -o tracks_erai_${VAR}.pickle
else
  python ../pystormtracker/stormtracker.py -i ${FILE} -v ${VAR} -o tracks_erai_${VAR}.pickle
fi
