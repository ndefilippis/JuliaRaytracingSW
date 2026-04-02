#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12GB
#SBATCH --time=1:00:00
#SBATCH --job-name=compute_b
#SBATCH --account=torch_pr_540_general
#SBATCH --array=1-11
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=two-layer-runs.txt

run_directory=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
fourier_directory=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

rundir=$SCRATCH/rsw_plots/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
output_directory=${rundir}/twoL-${SLURM_ARRAY_TASK_ID}.jld2

mkdir -p $rundir
cp compute_b_parameter.jl $rundir
cp ../AnalysisUtils.jl $rundir

cd $rundir

julia -t $SLURM_CPUS_PER_TASK compute_b_parameter.jl $run_directory $fourier_directory $output_directory > run.log

exit
