#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=05:00:00
#SBATCH --job-name=rsw-plots
#SBATCH --array=1-30
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=rsw-runs.txt

directory=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
tag=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

rundir=$SCRATCH/rsw_plots/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}

mkdir -p $rundir
cp RSWAnalysisDriver.jl RSWAnalysisSuite.jl RSWEnergetics.jl RSWTransferFunction.jl $rundir
cp ../AnalysisUtils.jl $rundir
cp ../../../rsw/RSWUtils.jl $rundir
cp ../../../utils/ExactRadialSpectrum.jl $rundir

cd $rundir
mkdir images

julia -t $SLURM_CPUS_PER_TASK RSWAnalysisDriver.jl $directory $tag > run.log

exit

