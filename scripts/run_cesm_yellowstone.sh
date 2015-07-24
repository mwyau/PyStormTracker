#!/bin/bash
#
# LSF batch script to run an MPI application
#
#BSUB -P STDD0004            # project code
#BSUB -W 00:30               # wall-clock time (hrs:mins)
#BSUB -n 16                  # number of tasks in job         
#BSUB -R "span[ptile=16]"    # run 16 MPI tasks per node
#BSUB -J cesm_16             # job name
#BSUB -o job_name.%J.out     # output file name in which %J is replaced by the job ID
#BSUB -e job_name.%J.err     # error file name in which %J is replaced by the job ID
#BSUB -q small               # queue

source /glade/u/apps/opt/lmod/4.2.1/init/bash

INPUT=/glade/p/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/hourly6/PSL/b.e11.B20TRC5CNBDRD.f09_g16.031.cam.h2.PSL.1990010100Z-2005123118Z.nc

mpirun.lsf python ../pystormtracker/stormtracker.py -i $INPUT -v PSL -o tracks_cesm.pickle
