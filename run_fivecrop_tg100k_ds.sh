#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=fivecrop_tg100k_ds
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/tg100k_ds-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/tg100k_ds-%j.err

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fivecrop_tgcompare.py \
    --root /work/u4546465/evaluation/tg_compare_100k/tg_compare_100k \
    --out_dir /home/u4546465/iqa_eval/results/tg_compare_100k \
    --device cuda
