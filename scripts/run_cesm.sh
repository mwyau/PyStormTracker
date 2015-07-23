#!/bin/bash
if [ "$1" != "" ]; then
  mpirun -np $1 python ../pystormtracker/stormtracker.py -i /Users/mwyau/Documents/CESM/b.e11.B20TRC5CNBDRD.f09_g16.031.cam.h2.PSL.1990010100Z-2005123118Z.nc -v PSL -o tracks_cesm.pickle
else
  python ../pystormtracker/stormtracker.py -i /Users/mwyau/Documents/CESM/b.e11.B20TRC5CNBDRD.f09_g16.031.cam.h2.PSL.1990010100Z-2005123118Z.nc -v PSL -o tracks_cesm.pickle
fi
