#!/bin/bash
#SBATCH --job-name=morph_step2_3
#SBATCH --account=pi_wanghongqiao
#SBATCH --partition=gpu4Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/public/home/hpc242111131/G-TAF/MORPH/logs/step2_3_%j.out
#SBATCH --error=/public/home/hpc242111131/G-TAF/MORPH/logs/step2_3_%j.err

mkdir -p /public/home/hpc242111131/G-TAF/MORPH/logs

module load CUDA/12.1.0
source /public/home/hpc242111131/miniconda3/etc/profile.d/conda.sh
conda activate morph_env

# 新格式：每场输出 shape_graph_nodes_{gid}.parquet + shape_graph_edges_{gid}.parquet
cd /public/home/hpc242111131/G-TAF/MORPH/General/scripts
python step2_3_batch_hpc.py --all --workers 16
