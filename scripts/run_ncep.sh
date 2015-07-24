#!/bin/bash
if [ "$1" != "" ]; then
  mpirun -np $1 python ../pystormtracker/stormtracker.py -i ../slp.2012.nc -v slp -o tracks_ncep.pickle
else
  python ../pystormtracker/stormtracker.py -i ../slp.2012.nc -v slp -o tracks_ncep.pickle
fi
