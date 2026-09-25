#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=fivecrop_perc_nods
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/perceptual_nods-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/perceptual_nods-%j.err

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fivecrop_tgcompare.py \
    --no_downsample \
    --root /work/u4546465/evaluation/Perceptual_gs \
    --ref_name reference.png \
    --out_dir /home/u4546465/iqa_eval/results/Perceptual_gs \
    --device cuda
