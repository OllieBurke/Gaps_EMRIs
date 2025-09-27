#!/bin/sh

#SBATCH --job-name=seed_0
#SBATCH --output=output_logs/output_%j.out
#SBATCH --error=error_logs/error_%j.err
#SBATCH --account=lisa
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_all
#SBATCH --mem=100G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=23:00:00

module load conda
module unload conda
module load conda
conda activate few_fisher 
module load cuda/12.4.1
python /home/ad/burkeol/work/Gaps_EMRIs/EMRIs/PE_simulations/mcmc_code/mcmc_run.py


