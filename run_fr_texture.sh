#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=fr_texture
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/fr_texture-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/fr_texture-%j.err

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fr_tgcompare.py \
    --root /work/u4546465/evaluation/tg_compare_100k/tg_compare_100k \
    --methods 2dgs_textured,2dgs_textured_crn_wd \
    --fid_csv /home/u4546465/iqa_eval/results/tg_compare_100k/tgcompare_per_scene.csv \
    --out_dir /home/u4546465/iqa_eval/results/tg_compare_texture \
    --device cuda
