#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64GB
#SBATCH --time=5:00:00
#SBATCH --job-name=thomasyamada_simulation

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/thomasyamada_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp cpu-driver.sbatch ThomasYamada.jl TYdriver.jl TYUtils.jl README.txt $rundir
cp cpu-setup/RestartParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TYdriver.jl CPU RESTART > run.log

exit
