#!/bin/bash
#SBATCH --account=mst108318
#SBATCH --job-name=fivecrop_wdrpw
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/u4546465/iqa_eval/logs/wdrpw-%j.out
#SBATCH --error=/home/u4546465/iqa_eval/logs/wdrpw-%j.err

source /etc/profile.d/*lmod* 2>/dev/null
module load miniconda3/26.1.1
source activate wd

cd /home/u4546465/iqa_eval
/home/u4546465/.conda/envs/wd/bin/python eval_fid_cmmd_fivecrop.py \
    --no_downsample \
    --cat3dgs_root /work/u4546465/evaluation/CAT3DGSPro_wdr_pw/CAT3DGSPro_organized \
    --methods 3DCR,Orig,wdr_pw \
    --out_dir /home/u4546465/iqa_eval/results/CAT3DGSPro_wdr_pw \
    --device cuda
