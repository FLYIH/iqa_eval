#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=fivecrop_hac_ds
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/hac_ds-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/hac_ds-%j.err

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fid_cmmd_fivecrop.py \
    --methods HACplus+CR \
    --out_dir /home/u4546465/iqa_eval/results/HACplus+CR \
    --device cuda
