#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=2-12:00:00
#SBATCH --account=torch_pr_540_general
#SBATCH --job-name=qgsw

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/swqg/$SLURM_JOB_ID
mkdir -p $rundir
cp swqg-driver.sbatch SWQG.jl SWQGDriver.jl SWQGMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp SWQGParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK SWQGMain.jl > run.log

exit
