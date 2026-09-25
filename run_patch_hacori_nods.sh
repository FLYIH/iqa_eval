#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=patch_hacori_nods
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/patch_hacori_nods-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/patch_hacori_nods-%j.err

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fid_cmmd_patch.py \
    --no_downsample \
    --methods HAC_ori \
    --out_dir /home/u4546465/iqa_eval/results/HAC_ori \
    --device cuda
