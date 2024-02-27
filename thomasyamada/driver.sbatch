#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=24GB
#SBATCH --time=12:00:00
#SBATCH --job-name=thomasyamada_simulation

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/thomasyamada_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp driver.sbatch Parameters.jl ThomasYamada.jl TYdriver.jl TYUtils.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TYdriver.jl CPU > run.log

exit
