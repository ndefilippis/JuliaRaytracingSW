#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=3GB
#SBATCH --time=00:10:00
#SBATCH --job-name=rsw_f
#SBATCH --array=257-768
#SBATCH --output=./slurm-output/slurm-%A_%a.out

# Split each job into 64 parts, analyze 12 rsw runs, 64*12=768
# Start at job 5, initial index = 4*64 + 1

module purge

export NUM_JULIA_THREADS=`nproc`

source multi-config.sh

task_size=$(($SLURM_ARRAY_TASK_MAX/$num_jobs))
job_index=$((($SLURM_ARRAY_TASK_ID-1)/$task_size+1))
partition_id=$((($SLURM_ARRAY_TASK_ID-1)%$task_size+1))
analysis_directory=${analysis_root_directory}/${job_index}

rundir=$SCRATCH/rsw_fourier/${SLURM_ARRAY_JOB_ID}/$job_index/$partition_id
max_file=$(($(find $analysis_directory -name rsw.[0-9]*.jld2 | wc -l)-1))
job_size=$(($max_K/$task_size))

mkdir -p $rundir
cp fourier-job.sbatch FourierRSW.jl README.txt $rundir
cp ../RSWUtils.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK FourierRSW.jl $analysis_directory $max_file $partition_id $job_size > run.log

exit
