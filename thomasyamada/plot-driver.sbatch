#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --job-name=plot_TY

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/thomasyamada_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp plot-driver.sbatch makeTYplot.jl TYUtils.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK makeTYplot.jl $1 > run.log

exit
