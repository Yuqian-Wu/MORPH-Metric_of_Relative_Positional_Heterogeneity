#!/bin/bash
#SBATCH --job-name=morph_step3_2_3
#SBATCH --account=pi_wanghongqiao
#SBATCH --partition=gpu4Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=30-00:00:00
#SBATCH --output=/public/home/hpc242111131/G-TAF/MORPH/logs/step3_2_3_%j.out
#SBATCH --error=/public/home/hpc242111131/G-TAF/MORPH/logs/step3_2_3_%j.err

mkdir -p /public/home/hpc242111131/G-TAF/MORPH/logs

module purge
module load anaconda3/4.9.2
module load CUDA/12.1.0
module load GNU/gcc-12.2.0
source activate morph_env

export MORPH_ENV=hpc
export PYTHONPATH="/public/home/hpc242111131/unravelsports-main (modified for 2022 WC):$PYTHONPATH"

cd /public/home/hpc242111131/G-TAF/MORPH/General
python scripts/step3_2_3_train.py \
    --all \
    --workers 8 \
    2>&1 | tee /public/home/hpc242111131/G-TAF/MORPH/logs/step3_2_3_$(date +%Y%m%d_%H%M%S).log
