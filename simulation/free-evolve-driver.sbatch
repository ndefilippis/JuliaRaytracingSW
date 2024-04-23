#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --job-name=2Lsim

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/twolayer_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp Driver.jl driver.sbatch $rundir
cp FreelyEvolvingSimulation.jl $rundir/TwoLayerSimulation.jl
cp FreelyEvolvingParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK Driver.jl > run.log

exit
