#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=fivecrop_tgsame
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/tgsame-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/tgsame-%j.err

source /etc/profile.d/*lmod* 2>/dev/null
module load miniconda3/26.1.1
source activate wd

cd /home/u4546465/iqa_eval
python eval_fivecrop_tgcompare.py \
    --no_downsample \
    --root /work/u4546465/evaluation/tg_compare_100k_samecount/tg_compare_100k_samecount \
    --out_dir /home/u4546465/iqa_eval/results/tg_compare_100k_samecount \
    --device cuda
