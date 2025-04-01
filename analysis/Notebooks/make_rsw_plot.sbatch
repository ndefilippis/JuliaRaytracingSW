#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=05:00:00
#SBATCH --job-name=make-plot

module purge

julia -t $SLURM_CPUS_PER_TASK MakeOmegaKPlots.jl

exit

