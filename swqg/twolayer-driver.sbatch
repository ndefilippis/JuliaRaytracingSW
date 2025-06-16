#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=1-05:00:00
#SBATCH --job-name=2Lqg

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$VAST/2Lqg/$SLURM_JOB_ID
mkdir -p $rundir
cp twolayer-driver.sbatch TwoLayerQG.jl TwoLayerDriver.jl TwoLayerMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp TwoLayerParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TwoLayerMain.jl > run.log

exit
