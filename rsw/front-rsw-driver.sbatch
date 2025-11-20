#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=6GB
#SBATCH --time=12:59:00
#SBATCH --job-name=rsw
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=nad9961@nyu.edu
#SBATCH --mail-type=BEGIN,FAIL

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$VAST/rsw/$SLURM_JOB_ID/

mkdir -p $rundir
cp .correct_e front-rsw-driver.sbatch RotatingShallowWater.jl RSWDriver.jl RSWMain.jl README.txt $rundir 
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp FrontParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RSWMain.jl > run.log

exit
