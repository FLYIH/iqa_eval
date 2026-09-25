#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=validate_patch
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/u4546465/iqa_eval/logs/validate_patch-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/validate_patch-%j.err

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fid_patch.py --validate_only \
    --cat3dgs_root /work/u4546465/evaluation/CAT3DGSPro_organized \
    --hac_root /work/u4546465/evaluation/HACplus+CR \
    --device cuda
