#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=23:59:59
#SBATCH --job-name=rsw
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=nad9961@nyu.edu
#SBATCH --mail-type=BEGIN

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$VAST/rsw/$SLURM_JOB_ID
mkdir -p $rundir
cp rsw-driver.sbatch RotatingShallowWater.jl RSWDriver.jl RSWMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp RSWParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RSWMain.jl > run.log

exit
