#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mrsw
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=nad9961@nyu.edu

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$VAST/mrsw/$SLURM_JOB_ID
mkdir -p $rundir
cp mrsw-driver.sbatch RSWDriver.jl RSWMain.jl README.txt $rundir
cp ModifiedShallowWater.jl $rundir/RotatingShallowWater.jl
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp RSWParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RSWMain.jl > run.log

exit
