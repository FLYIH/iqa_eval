#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=fivecrop_tg100k
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/tg100k-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/tg100k-%j.err

source /etc/profile.d/*lmod* 2>/dev/null
module load miniconda3/26.1.1
source activate wd

cd /home/u4546465/iqa_eval
python eval_fivecrop_tgcompare.py \
    --no_downsample \
    --root /work/u4546465/evaluation/tg_compare_100k/tg_compare_100k \
    --out_dir /home/u4546465/iqa_eval/results/tg_compare_100k \
    --device cuda
