#!/bin/bash
#SBATCH --job-name=morph_step1_4
#SBATCH --account=pi_wanghongqiao
#SBATCH --partition=gpu4Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/hpc242111131/G-TAF/MORPH/logs/step1_4_%j.out
#SBATCH --error=/public/home/hpc242111131/G-TAF/MORPH/logs/step1_4_%j.err

mkdir -p /public/home/hpc242111131/G-TAF/MORPH/logs

module purge
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate morph_env

export MORPH_ENV=hpc

cd /public/home/hpc242111131/G-TAF/MORPH/General/scripts
python step1_4_scaling.py --all \
    2>&1 | tee /public/home/hpc242111131/G-TAF/MORPH/logs/step1_4_$(date +%Y%m%d_%H%M%S).log
