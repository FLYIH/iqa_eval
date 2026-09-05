#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=fivecrop_nods
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/fivecrop_nodownsample-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/fivecrop_nodownsample-%j.err

source /etc/profile.d/*lmod* 2>/dev/null
module load miniconda3/26.1.1
source activate wd

cd /home/u4546465/iqa_eval
python eval_fid_cmmd_fivecrop.py \
    --no_downsample \
    --out_dir /home/u4546465/iqa_eval/results \
    --device cuda
