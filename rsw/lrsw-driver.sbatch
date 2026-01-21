#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=6GB
#SBATCH --time=1-23:59:59
#SBATCH --job-name=lrsw
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=nad9961@nyu.edu

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$VAST/lrsw/$SLURM_JOB_ID
mkdir -p $rundir
cp .correct_e lrsw-driver.sbatch RSWDriver.jl RSWMain.jl README.txt $rundir
cp LinborgShallowWater.jl $rundir/RotatingShallowWater.jl
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp RSWParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RSWMain.jl > run.log

exit
