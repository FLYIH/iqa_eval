#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=patch_wdrpw_nods
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/patch_wdrpw_nods-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/patch_wdrpw_nods-%j.err

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fid_cmmd_patch.py \
    --no_downsample \
    --cat3dgs_root /work/u4546465/evaluation/CAT3DGSPro_wdr_pw/CAT3DGSPro_organized \
    --methods 3DCR,Orig,wdr_pw \
    --out_dir /home/u4546465/iqa_eval/results/CAT3DGSPro_wdr_pw \
    --device cuda
