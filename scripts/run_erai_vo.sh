#!/bin/bash
FILE=/array1/Reanalysis/ERA-Interim/vort.850.075.nc
VAR=vo
if [ "$1" != "" ]; then
  mpirun -np $1 python ../pystormtracker/stormtracker.py -i ${FILE} -v ${VAR} -o tracks_erai_${VAR}.pickle -m max
else
  python ../pystormtracker/stormtracker.py -i ${FILE} -v ${VAR} -o tracks_erai_${VAR}.pickle -m max
fi
