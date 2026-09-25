#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=patch_wdr_nods
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/patch_wdr_nods-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/patch_wdr_nods-%j.err

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fid_cmmd_patch_wdr.py \
    --no_downsample \
    --out_dir /home/u4546465/iqa_eval/results/wdr \
    --device cuda
