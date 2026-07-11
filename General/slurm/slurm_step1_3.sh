#!/bin/bash
#SBATCH --job-name=morph_step1_3
#SBATCH --account=pi_wanghongqiao
#SBATCH --partition=gpu4Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/hpc242111131/G-TAF/MORPH/logs/step1_3_%j.out
#SBATCH --error=/public/home/hpc242111131/G-TAF/MORPH/logs/step1_3_%j.err

mkdir -p /public/home/hpc242111131/G-TAF/MORPH/logs

module load CUDA/12.1.0
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate morph_env

cd /public/home/hpc242111131/G-TAF/MORPH/General/scripts
python step1_3_tactical_intent.py --all --batch_size 256
